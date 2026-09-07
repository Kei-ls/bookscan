"""Document-Scanner 가중치를 내 사진으로 이어 학습한다.

왜 파인튜닝인가
---------------
남이 올린 이 체크포인트는 "장면 속 문서"를 찾도록 배웠다. 펼친 책의 **한 쪽**은
그 분포 밖이다 — 책등 너머에도 같은 종이가 이어지니, 모델 입장에서 거기 선을
그을 이유가 없다. 실제로 이 저장소가 부딪힌 실패가 정확히 그것이다. 책등 쪽
경계는 국소 기하로는 못 찾는다(경계 안쪽 15px에 잉크가 0.4%뿐이고, 바깥은
그림자가 매끄럽게 이어진다). 규칙으로 짤 수 없고 보여줘서 가르쳐야 한다.

두 가지를 같이 바꾼다
---------------------
**해상도.** 원본 체크포인트는 256×256을 받는다. 4284×5712 사진이면 한 칸이 원본
22픽셀이라, 책등 선을 몇 픽셀 안에서 놓아야 하는 일에는 원리적으로 모자란다.
기본값을 384로 올린다. U-Net은 5번 풀링하므로 32의 배수여야 한다. 다만 해상도만
올리고 학습하지 않으면 (1,9)/(9,1) 스트립 합성곱의 수용영역이 상대적으로 좁아져
오히려 나빠진다. 그래서 둘은 세트다.

**확률 분포.** 원본 체크포인트는 확률을 0~1로 벌리지 못하고 0.47~0.72 안에
뭉쳐 놓는다(`runtime._binarize` 참고 — 그래서 고정 문턱 0.5를 못 쓰고 상대
위치로 자르는 우회가 들어가 있다). BCE로 이어 학습하면 그 눌림이 펴진다.

자료
----
`tools/export_labels.py`가 만든 폴더를 먹는다. `<이름>.jpg` + `<이름>_mask.png`
쌍이면 되므로 `tests/synthetic`도 그대로 넣을 수 있다(연습용으로 쓸 만하다).

사용:
    python tools/export_labels.py
    python tools/finetune.py labels --epochs 40
    python tools/finetune.py tests/synthetic --epochs 2 --smoke   # 동작 확인만
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bookscan.models.catalog import DOCSCANNER, FINETUNED, WEIGHTS_DIR  # noqa: E402
from bookscan.models.docscanner import build_model  # noqa: E402

#: 검증용으로 떼어 둘 비율. 자료가 적을 때는 최소 한 장은 뗀다.
VAL_FRACTION = 0.2
#: 증강 폭 — 회전(도), 배율, 밝기(0~255), 대비
AUG_ROTATE, AUG_SCALE, AUG_BRIGHT, AUG_CONTRAST = 7.0, 0.12, 25.0, 0.20


def find_pairs(root: Path) -> list[tuple[Path, Path]]:
    """(사진, 마스크) 쌍. `<이름>_mask.png` 를 기준으로 짝을 찾는다."""
    pairs = []
    for mask in sorted(root.glob("*_mask.png")):
        stem = mask.name[: -len("_mask.png")]
        for ext in (".jpg", ".jpeg", ".png"):
            img = mask.with_name(stem + ext)
            if img.exists():
                pairs.append((img, mask))
                break
    return pairs


def _augment(img: np.ndarray, mask: np.ndarray,
             rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """가벼운 증강. 사진을 크게 흔들지 않는다 — 자료가 적을수록 원본에 가까운
    변형이 안전하고, 우리가 가르치려는 건 색이 아니라 **경계의 위치**다."""
    h, w = img.shape[:2]
    ang = rng.uniform(-AUG_ROTATE, AUG_ROTATE)
    sc = 1.0 + rng.uniform(-AUG_SCALE, AUG_SCALE)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
    mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    a = 1.0 + rng.uniform(-AUG_CONTRAST, AUG_CONTRAST)
    b = rng.uniform(-AUG_BRIGHT, AUG_BRIGHT)
    return np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8), mask


def _prepare(img_path: Path, mask_path: Path, size: tuple[int, int],
             rng: random.Random | None) -> tuple[np.ndarray, np.ndarray]:
    """한 쌍을 학습 입력으로. 정규화는 `runtime.segment`와 똑같이 맞춰야 한다 —
    학습할 때와 추론할 때 입력 분포가 다르면 학습한 것이 그대로 안 나온다."""
    gray = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imdecode(np.fromfile(mask_path, np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None or mask is None:
        raise RuntimeError(f"읽지 못함: {img_path.name}")
    if rng is not None:
        gray, mask = _augment(gray, mask, rng)
    x = cv2.resize(gray, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    y = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    std = float(x.std())
    x = (x - x.mean()) / (std if std > 1e-6 else 1.0)
    return x, (y > 127).astype(np.float32)


def _batches(pairs, size, batch, rng, torch):
    order = list(range(len(pairs)))
    if rng is not None:
        rng.shuffle(order)
    for i in range(0, len(order), batch):
        xs, ys = [], []
        for j in order[i:i + batch]:
            x, y = _prepare(pairs[j][0], pairs[j][1], size, rng)
            xs.append(x)
            ys.append(y)
        yield (torch.from_numpy(np.stack(xs))[:, None],
               torch.from_numpy(np.stack(ys))[:, None])


def _iou(logits, target, torch) -> float:
    pred = (torch.sigmoid(logits) > 0.5).float()
    inter = (pred * target).sum().item()
    union = pred.sum().item() + target.sum().item() - inter
    return inter / union if union > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("labels", help="export_labels.py 가 만든 폴더")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--size", type=int, default=FINETUNED.input_size[0],
                    help="입력 한 변 (32의 배수)")
    ap.add_argument("--out", default=str(WEIGHTS_DIR / FINETUNED.weights_name))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="동작만 확인한다 (가중치를 저장하지 않는다)")
    args = ap.parse_args()

    if args.size % 32:
        print(f"입력 크기는 32의 배수여야 합니다 (U-Net이 5번 풀링합니다): {args.size}")
        return 1

    root = Path(args.labels)
    if not root.is_absolute():
        root = Path(__file__).resolve().parent.parent / root
    pairs = find_pairs(root)
    if not pairs:
        print(f"{root} 에 (사진, 마스크) 쌍이 없습니다. "
              f"먼저 python tools/export_labels.py 를 돌리세요.")
        return 1

    base = WEIGHTS_DIR / DOCSCANNER.weights_name
    if not base.exists():
        print(f"바탕이 될 가중치가 없습니다: {base}\n"
              f"UI의 '검출 모델 > 설치'로 먼저 내려받으세요.")
        return 1

    import torch  # 여기서만 필요한 선택 의존성

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    idx = list(range(len(pairs)))
    rng.shuffle(idx)
    n_val = max(1, int(len(pairs) * VAL_FRACTION)) if len(pairs) > 1 else 0
    val = [pairs[i] for i in idx[:n_val]]
    train = [pairs[i] for i in idx[n_val:]]
    size = (args.size, args.size)

    print(f"자료 {len(pairs)}쌍 — 학습 {len(train)}, 검증 {len(val)}  |  입력 {args.size}px")
    if len(train) < 20 and not args.smoke:
        print("⚠ 자료가 20쌍도 안 됩니다. 이 정도로는 모델이 이 몇 장을 외울 뿐이라"
              "\n  다른 사진에서 더 나빠질 수 있습니다. 200~500쌍을 목표로 모으세요.")

    model = build_model(torch)
    state = torch.load(base, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.train()
    torch.set_grad_enabled(True)  # runtime._load 가 전역으로 꺼 둘 수 있다

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    bce = torch.nn.BCEWithLogitsLoss()

    def dice(logits, target):
        p = torch.sigmoid(logits)
        num = 2 * (p * target).sum() + 1.0
        den = p.sum() + target.sum() + 1.0
        return 1.0 - num / den

    best = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, total, seen = time.time(), 0.0, 0
        for x, y in _batches(train, size, args.batch, rng, torch):
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = bce(out, y) + dice(out, y)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(x)
            seen += len(x)
        sched.step()

        line = f"epoch {ep:3d}/{args.epochs}  loss {total / max(seen, 1):.4f}"
        if val:
            model.eval()
            with torch.no_grad():
                ious = [_iou(model(x), y, torch)
                        for x, y in _batches(val, size, args.batch, None, torch)]
            miou = float(np.mean(ious)) if ious else 0.0
            line += f"  val IoU {miou:.4f}"
            if miou > best and not args.smoke:
                best = miou
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), args.out)
                line += "  ← 저장"
        print(line + f"  ({time.time() - t0:.1f}s)")

    if args.smoke:
        print("\n동작 확인만 했습니다 (가중치를 저장하지 않았습니다).")
        return 0
    if best < 0:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.out)
        print(f"\n검증 자료가 없어 마지막 상태를 저장했습니다 → {args.out}")
    else:
        print(f"\n가장 좋았던 상태 (val IoU {best:.4f}) → {args.out}")
    print(f"입력 {args.size}px 로 학습했습니다. `bookscan/models/catalog.py`의 "
          f"FINETUNED.input_size 와 같아야 합니다 (지금 {FINETUNED.input_size[0]}px).")
    print("UI의 `검출 모델`에 '파인튜닝'이 새로 뜹니다. "
          "내장 알고리즘과 견줘 보고 나은 쪽을 쓰세요.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
