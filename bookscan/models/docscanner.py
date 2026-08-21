"""Document-Scanner U-Net — 문서 영역 분할 신경망.

출처: https://github.com/Saeed-Biabani/Document-Scanner (MIT, (c) 2024 Saeed-Biabani)
가중치: https://huggingface.co/ordaktaktak/Document-Scanner

원 저장소의 구조를 그대로 옮겼다. 학습된 가중치를 그대로 불러와야 하므로
**모듈 속성 이름을 바꾸면 안 된다** — state_dict의 키가 속성 이름에서 나온다.
(예: ``encoder_seq.0.spatial_conv_1.layer.0.weight``)

구조 요약
  · 인코더는 (1,9)와 (9,1) 두 방향의 긴 스트립 합성곱을 따로 돌려 이어붙인다.
    문서의 긴 직선 모서리를 잡기 좋은 형태다.
  · 디코더는 전치 합성곱으로 올리면서 스킵 연결을 이어붙이는 평범한 U-Net.
  · 출력은 1채널 로짓. 시그모이드를 씌우면 "여기가 문서다" 확률이 된다.
"""

from __future__ import annotations

from typing import Any

#: 원 저장소 Structure/getConfig.py 의 config_ 와 같아야 한다.
INPUT_SIZE = (256, 256)
ENC_CHANNELS = [1, 64, 128, 256, 512, 512]
DEC_CHANNELS = [512, 512, 256, 128, 64, 1]


def build_model(torch: Any) -> Any:
    """torch를 인자로 받아 모델 클래스를 정의하고 인스턴스를 돌려준다.

    torch는 선택 의존성이라 모듈 최상단에서 import할 수 없다. 그래서 클래스
    정의 자체를 함수 안에 둔다. 조금 특이하지만, 이 파일을 torch 없이도
    import할 수 있게 해준다 (모델 목록을 보여주려면 import는 돼야 한다).
    """
    nn = torch.nn
    F = torch.nn.functional

    class ConvLayer(nn.Module):
        def __init__(self, in_ch, out_ch, kernel=(3, 3), stride=1, padding="same"):
            super().__init__()
            self.layer = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.layer(x)

    class Encoder(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.spatial_conv_1 = ConvLayer(in_ch, out_ch, (1, 9))
            self.spatial_conv_2 = ConvLayer(in_ch, out_ch, (9, 1))
            self.out = ConvLayer(out_ch * 2, out_ch)

        def forward(self, x):
            return self.out(torch.concat((self.spatial_conv_1(x), self.spatial_conv_2(x)), dim=1))

    class Decoder(nn.Module):
        def __init__(self, in_ch, out_ch, mid_ch):
            super().__init__()
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=3, stride=2,
                                         padding=1, bias=False)
            self.conv = nn.Sequential(ConvLayer(mid_ch, out_ch), ConvLayer(out_ch, out_ch))

        def forward(self, x, skip):
            x = self.up(x)
            # 전치 합성곱은 홀수 크기를 내놓기 때문에(8 -> 15) 스킵과 한 픽셀씩
            # 어긋난다. 양쪽으로 나눠 패딩해 맞춘다.
            dy = skip.size()[2] - x.size()[2]
            dx = skip.size()[3] - x.size()[3]
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
            return self.conv(torch.concat((x, skip), 1))

    class Detector(nn.Module):
        def __init__(self, enc_channels, dec_channels):
            super().__init__()
            self.encoder_seq = nn.ModuleList(
                Encoder(enc_channels[i], enc_channels[i + 1])
                for i in range(len(enc_channels) - 1)
            )
            self.decoder_seq = nn.ModuleList(
                Decoder(dec_channels[i], dec_channels[i + 1],
                        dec_channels[i] + dec_channels[i] // 2)
                for i in range(len(dec_channels) - 1)
            )
            self.out = nn.Conv2d(1, 1, 1, padding="same")

        def forward(self, x):
            skips = []
            for layer in self.encoder_seq:
                x = layer(x)
                skips.append(x)
                x = F.avg_pool2d(x, 2)
            for layer, skip in zip(self.decoder_seq, skips[::-1]):
                x = layer(x, skip)
            return self.out(x)

    return Detector(ENC_CHANNELS, DEC_CHANNELS)
