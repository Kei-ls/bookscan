/* Bookscan 프론트엔드 — 빌드 스텝 없는 바닐라 JS */
"use strict";

const $ = (sel) => document.querySelector(sel);
const state = {
  pages: [],
  sources: {},
  project: "",
  dir: "",
  editing: null,   // { sid, corners, imgW, imgH, settings }
  dragId: null,
  cameraOn: false,
  models: [],
  selectedModel: "classic",
};

// ------------------------------------------------------------------ helpers
async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: opts.body instanceof FormData ? {} : { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    let msg = `${res.status}`;
    try { msg = (await res.json()).detail || msg; } catch { /* 본문 없음 */ }
    throw new Error(msg);
  }
  return res.status === 204 ? null : res.json();
}

let toastTimer;
function toast(msg, kind = "") {
  const el = $("#toast");
  el.textContent = msg;
  el.className = "toast " + kind;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, kind === "bad" ? 6000 : 3200);
}

function applySnapshot(data) {
  if (!data) return;
  if (data.pages) state.pages = data.pages;
  if (data.sources) state.sources = data.sources;
  if (data.project) state.project = data.project;
  if (data.dir) state.dir = data.dir;
  if (data.defaults && data.defaults.detector) state.selectedModel = data.defaults.detector;
  renderGrid();
}

// ------------------------------------------------------------------ 그리드
function renderGrid() {
  const grid = $("#grid");
  grid.innerHTML = "";
  $("#empty").hidden = state.pages.length > 0;

  for (const page of state.pages) {
    const card = document.createElement("div");
    card.className = "card";
    card.draggable = true;
    card.dataset.id = page.id;

    const lowConf = page.confidence > 0 && page.confidence < 0.6;
    card.innerHTML = `
      <img class="thumb" src="/api/pages/${page.id}/thumb?v=${page.version}" loading="lazy" alt=""
           draggable="false">
      ${lowConf ? `<span class="badge">확인 필요</span>` : ""}
      ${page.half ? `<span class="half">${page.half === "left" ? "왼쪽" : "오른쪽"}</span>` : ""}
      <div class="tools">
        <button data-act="left" title="앞으로 이동">◀</button>
        <button data-act="right" title="뒤로 이동">▶</button>
        <button data-act="edit" title="편집">편집</button>
        <button data-act="del" title="삭제">삭제</button>
      </div>
      <div class="meta">
        <span class="num">${page.index}</span>
        <span class="name" title="${page.filename}">${page.filename}</span>
      </div>`;

    card.querySelector('[data-act="left"]').onclick = (e) => { e.stopPropagation(); movePage(page.id, -1); };
    card.querySelector('[data-act="right"]').onclick = (e) => { e.stopPropagation(); movePage(page.id, +1); };
    card.querySelector('[data-act="edit"]').onclick = (e) => { e.stopPropagation(); openEditor(page); };
    card.querySelector('[data-act="del"]').onclick = async (e) => {
      e.stopPropagation();
      try { applySnapshot(await api(`/api/pages/${page.id}`, { method: "DELETE" })); }
      catch (err) { toast("삭제 실패: " + err.message, "bad"); }
    };
    card.ondblclick = () => openEditor(page);
    grid.appendChild(card);
  }
  attachDrag();
}

/** 한 칸 앞/뒤로 이동. 그리드가 여러 줄로 접힐 때는 드래그보다 이게 정확하다. */
async function movePage(id, delta) {
  const order = state.pages.map((p) => p.id);
  const i = order.indexOf(id);
  const j = i + delta;
  if (i < 0 || j < 0 || j >= order.length) return;
  [order[i], order[j]] = [order[j], order[i]];
  try { applySnapshot(await api("/api/pages/reorder", { method: "POST", body: JSON.stringify({ order }) })); }
  catch (err) { toast("순서 변경 실패: " + err.message, "bad"); }
}

function attachDrag() {
  const cards = document.querySelectorAll(".card");
  cards.forEach((card) => {
    card.addEventListener("dragstart", () => {
      state.dragId = card.dataset.id;
      card.classList.add("dragging");
    });
    card.addEventListener("dragend", () => {
      card.classList.remove("dragging");
      cards.forEach((c) => c.classList.remove("dropbefore", "dropafter"));
      state.dragId = null;
    });
    card.addEventListener("dragover", (e) => {
      e.preventDefault();
      if (!state.dragId || card.dataset.id === state.dragId) return;
      const r = card.getBoundingClientRect();
      const after = e.clientX > r.left + r.width / 2;
      card.classList.toggle("dropafter", after);
      card.classList.toggle("dropbefore", !after);
    });
    card.addEventListener("dragleave", () => card.classList.remove("dropbefore", "dropafter"));
    card.addEventListener("drop", async (e) => {
      e.preventDefault();
      e.stopPropagation();
      card.classList.remove("dropbefore", "dropafter");
      const dragged = state.dragId;
      if (!dragged || card.dataset.id === dragged) return;
      const r = card.getBoundingClientRect();
      const after = e.clientX > r.left + r.width / 2;

      const order = state.pages.map((p) => p.id).filter((id) => id !== dragged);
      let at = order.indexOf(card.dataset.id);
      if (after) at += 1;
      order.splice(at, 0, dragged);
      try { applySnapshot(await api("/api/pages/reorder", { method: "POST", body: JSON.stringify({ order }) })); }
      catch (err) { toast("순서 변경 실패: " + err.message, "bad"); }
    });
  });
}

// ------------------------------------------------------------------ 편집기
const canvas = $("#cornerCanvas");
const ctx = canvas.getContext("2d");
let editImg = null;
let dragCorner = -1;

function openEditor(page) {
  const src = state.sources[page.source];
  if (!src) return;
  const st = JSON.parse(JSON.stringify(src.settings));
  state.editing = {
    sid: page.source, pageId: page.id, settings: st,
    corners: (st.corners || src.corners || []).map((c) => c.slice()),
    outline: (src.outline || []).map((c) => c.slice()),
  };

  $("#editTitle").textContent = `편집 — ${src.filename}`;
  $("#editPreview").src = `/api/pages/${page.id}/full?v=${page.version}`;
  const noteLines = [...(src.notes || []), ...(page.notes || [])];
  $("#editNotes").textContent = noteLines.join("\n");

  syncControls();

  editImg = new Image();
  editImg.onload = () => {
    state.editing.imgW = editImg.naturalWidth;
    state.editing.imgH = editImg.naturalHeight;
    fitCanvas();
    drawCanvas();
  };
  editImg.src = `/api/sources/${page.source}/original`;
  $("#editor").hidden = false;
}

function syncControls() {
  const st = state.editing.settings;
  document.querySelectorAll("#modeSeg button").forEach((b) =>
    b.classList.toggle("on", b.dataset.mode === st.mode));
  document.querySelectorAll("#spreadSeg button").forEach((b) =>
    b.classList.toggle("on", b.dataset.spread === st.spread));
  document.querySelectorAll("#spineSeg button").forEach((b) =>
    b.classList.toggle("on", b.dataset.spine === (st.spine || "auto")));
  $("#autoRotateToggle").checked = st.auto_rotate !== false;
  $("#trimToggle").checked = st.trim !== false;
  syncDetectorSelect(st.detector || "classic");
  $("#curveToggle").checked = st.curve !== false;
  $("#paperSelect").value = st.paper || "auto";
  $("#dewarpToggle").checked = !!st.dewarp;
  $("#brightRange").value = st.brightness || 0;
  $("#brightVal").textContent = st.brightness || 0;
  $("#contrastRange").value = st.contrast || 0;
  $("#contrastVal").textContent = st.contrast || 0;

  const src = state.sources[state.editing.sid];
  const g = st.gutter != null ? st.gutter : (src && src.gutter != null ? src.gutter : 0.5);
  $("#gutterRange").value = Math.round(g * 100);
  $("#gutterVal").textContent = Math.round(g * 100) + "%";
  $("#gutterField").style.display = st.spread === "single" ? "none" : "";
}

function fitCanvas() {
  const wrap = canvas.parentElement;
  const maxW = Math.max(320, wrap.clientWidth - 4);
  const maxH = Math.max(280, window.innerHeight * 0.62);
  const s = Math.min(maxW / state.editing.imgW, maxH / state.editing.imgH, 1);
  canvas.width = Math.round(state.editing.imgW * s);
  canvas.height = Math.round(state.editing.imgH * s);
  state.editing.scale = s;
}

function drawCanvas() {
  if (!editImg || !state.editing) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(editImg, 0, 0, canvas.width, canvas.height);

  const pts = state.editing.corners.map(([x, y]) => [x * state.editing.scale, y * state.editing.scale]);
  if (pts.length !== 4) return;

  ctx.save();
  ctx.beginPath();
  ctx.moveTo(pts[0][0], pts[0][1]);
  pts.slice(1).forEach(([x, y]) => ctx.lineTo(x, y));
  ctx.closePath();
  ctx.fillStyle = "rgba(91,156,255,.14)";
  ctx.fill();
  ctx.strokeStyle = "#5b9cff";
  ctx.lineWidth = 2;
  ctx.stroke();

  // 굽은 테두리를 실제로 폈다면 그 곡선을 겹쳐 보여준다. 파란 사각형과 노란
  // 곡선이 벌어진 만큼이 사각형으로는 담지 못하던 부분이다.
  const outline = state.editing.outline || [];
  if (outline.length > 3) {
    ctx.beginPath();
    outline.forEach(([x, y], i) => {
      const cx = x * state.editing.scale, cy = y * state.editing.scale;
      if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy);
    });
    ctx.closePath();
    ctx.strokeStyle = "#ffd23f";
    ctx.lineWidth = 2.5;
    ctx.stroke();
  }

  pts.forEach(([x, y], i) => {
    ctx.beginPath();
    ctx.arc(x, y, 9, 0, Math.PI * 2);
    ctx.fillStyle = i === dragCorner ? "#fff" : "#5b9cff";
    ctx.fill();
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = "#fff";
    ctx.stroke();
  });
  ctx.restore();
}

function canvasPos(e) {
  const r = canvas.getBoundingClientRect();
  return [
    (e.clientX - r.left) * (canvas.width / r.width) / state.editing.scale,
    (e.clientY - r.top) * (canvas.height / r.height) / state.editing.scale,
  ];
}

canvas.addEventListener("pointerdown", (e) => {
  if (!state.editing || state.editing.corners.length !== 4) return;
  const [x, y] = canvasPos(e);
  let best = -1, bestD = Infinity;
  state.editing.corners.forEach(([cx, cy], i) => {
    const d = Math.hypot(cx - x, cy - y);
    if (d < bestD) { bestD = d; best = i; }
  });
  // 화면상 20px 이내를 잡은 것으로 친다
  if (bestD * state.editing.scale < 22) {
    dragCorner = best;
    canvas.setPointerCapture(e.pointerId);
    drawCanvas();
  }
});

canvas.addEventListener("pointermove", (e) => {
  if (dragCorner < 0 || !state.editing) return;
  const [x, y] = canvasPos(e);
  state.editing.corners[dragCorner] = [
    Math.max(0, Math.min(state.editing.imgW, x)),
    Math.max(0, Math.min(state.editing.imgH, y)),
  ];
  drawCanvas();
});

["pointerup", "pointercancel"].forEach((ev) =>
  canvas.addEventListener(ev, () => { dragCorner = -1; drawCanvas(); }));

// 편집기 컨트롤
document.querySelectorAll("#modeSeg button").forEach((b) => b.onclick = () => {
  state.editing.settings.mode = b.dataset.mode; syncControls();
});
document.querySelectorAll("#spineSeg button").forEach((b) => b.onclick = () => {
  state.editing.settings.spine = b.dataset.spine; syncControls();
});

document.querySelectorAll("#spreadSeg button").forEach((b) => b.onclick = () => {
  state.editing.settings.spread = b.dataset.spread; syncControls();
});
document.querySelectorAll("[data-rotate]").forEach((b) => b.onclick = () => {
  const st = state.editing.settings;
  st.rotate = ((st.rotate || 0) + parseInt(b.dataset.rotate, 10) + 360) % 360;
  toast(`회전 ${st.rotate}° — 적용을 눌러야 반영됩니다`);
});
$("#autoRotateToggle").onchange = (e) => {
  state.editing.settings.auto_rotate = e.target.checked;
};
$("#trimToggle").onchange = (e) => { state.editing.settings.trim = e.target.checked; };
$("#dewarpToggle").onchange = (e) => { state.editing.settings.dewarp = e.target.checked; };
$("#curveToggle").onchange = (e) => { state.editing.settings.curve = e.target.checked; };
$("#paperSelect").onchange = (e) => { state.editing.settings.paper = e.target.value; };
$("#brightRange").oninput = (e) => {
  state.editing.settings.brightness = +e.target.value; $("#brightVal").textContent = e.target.value;
};
$("#contrastRange").oninput = (e) => {
  state.editing.settings.contrast = +e.target.value; $("#contrastVal").textContent = e.target.value;
};
$("#gutterRange").oninput = (e) => {
  state.editing.settings.gutter = +e.target.value / 100;
  $("#gutterVal").textContent = e.target.value + "%";
};

$("#resetCorners").onclick = async () => {
  const btn = $("#resetCorners");
  btn.disabled = true;
  try {
    const data = await api(`/api/sources/${state.editing.sid}/reprocess`, {
      method: "POST", body: JSON.stringify({ reset_corners: true }),
    });
    applySnapshot(data);
    const src = state.sources[state.editing.sid];
    state.editing.corners = (src.corners || []).map((c) => c.slice());
    state.editing.settings.corners = null;
    drawCanvas();
    refreshEditorPreview();
    toast("자동 재검출 완료", "good");
  } catch (err) { toast("재검출 실패: " + err.message, "bad"); }
  finally { btn.disabled = false; }
};

$("#applyEdit").onclick = async () => {
  const btn = $("#applyEdit");
  btn.disabled = true;
  btn.textContent = "처리 중…";
  const st = state.editing.settings;
  try {
    const data = await api(`/api/sources/${state.editing.sid}/reprocess`, {
      method: "POST",
      body: JSON.stringify({
        corners: state.editing.corners.length === 4 ? state.editing.corners : null,
        mode: st.mode, rotate: st.rotate, auto_rotate: st.auto_rotate,
        dewarp: st.dewarp, curve: st.curve, trim: st.trim, detector: st.detector,
        spread: st.spread, spine: st.spine || "auto",
        gutter: st.spread === "single" ? null : st.gutter,
        paper: st.paper || "auto",
        brightness: st.brightness, contrast: st.contrast,
      }),
    });
    applySnapshot(data);
    refreshEditorPreview();
    toast("적용 완료", "good");
  } catch (err) { toast("적용 실패: " + err.message, "bad"); }
  finally { btn.disabled = false; btn.textContent = "적용"; }
};

function refreshEditorPreview() {
  const sid = state.editing && state.editing.sid;
  if (!sid) return;
  const page = state.pages.find((p) => p.source === sid);
  state.editing.outline = ((state.sources[sid] || {}).outline || []).map((c) => c.slice());
  drawCanvas();
  if (page) {
    $("#editPreview").src = `/api/pages/${page.id}/full?v=${page.version}&t=${Date.now()}`;
    const src = state.sources[sid];
    $("#editNotes").textContent = [...(src.notes || []), ...(page.notes || [])].join("\n");
    state.editing.pageId = page.id;
  }
}

// ------------------------------------------------------------------ 가져오기
$("#importFolder").onclick = async () => {
  const folder = $("#folderInput").value.trim();
  if (!folder) { toast("폴더 경로를 입력하세요.", "bad"); return; }
  try {
    const r = await api("/api/import/folder", { method: "POST", body: JSON.stringify({ folder }) });
    toast(`${r.imported}장 가져오는 중…`);
  } catch (err) { toast("가져오기 실패: " + err.message, "bad"); }
};

// 축(가로/세로)은 자동으로 맞지만 상하는 판별하지 않는다(orient.py 참고).
// 한 번에 찍은 묶음은 상하가 모두 같으므로 통째로 뒤집는 편이 빠르다.
$("#rotateAll").onclick = async () => {
  const btn = $("#rotateAll");
  if (!state.pages.length) { toast("먼저 사진을 가져오세요.", "bad"); return; }
  btn.disabled = true;
  try {
    const r = await api("/api/sources/rotate_all", {
      method: "POST", body: JSON.stringify({ delta: 180 }),
    });
    toast(`${r.rotated}장 뒤집는 중…`);
  } catch (err) { toast("회전 실패: " + err.message, "bad"); }
  finally { btn.disabled = false; }
};

$("#pickFiles").onclick = () => $("#fileInput").click();
$("#fileInput").onchange = (e) => uploadFiles([...e.target.files]);

async function uploadFiles(files) {
  const images = files.filter((f) => f.type.startsWith("image/") || /\.(hei[cf]|jpe?g|png|tiff?|bmp|webp)$/i.test(f.name));
  if (!images.length) { toast("이미지 파일이 없습니다.", "bad"); return; }
  const fd = new FormData();
  images.forEach((f) => fd.append("files", f));
  try {
    const r = await api("/api/import/upload", { method: "POST", body: fd });
    toast(`${r.imported}장 가져오는 중…`);
  } catch (err) { toast("업로드 실패: " + err.message, "bad"); }
}

// 파일 드래그 앤 드롭.
// 카드 순서를 바꾸는 중(state.dragId)에는 무시해야 한다. 그러지 않으면 썸네일을
// 끌 때 브라우저가 그 이미지를 파일로 취급해서 자기 자신을 다시 업로드해버린다.
let dragDepth = 0;
const isFileDrag = (e) => !state.dragId && [...e.dataTransfer.types].includes("Files");

window.addEventListener("dragenter", (e) => {
  if (!isFileDrag(e)) return;
  dragDepth++; $("#dropzone").hidden = false;
});
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("dragleave", () => { if (--dragDepth <= 0) { dragDepth = 0; $("#dropzone").hidden = true; } });
window.addEventListener("drop", (e) => {
  dragDepth = 0; $("#dropzone").hidden = true;
  if (!isFileDrag(e)) return;
  e.preventDefault();
  uploadFiles([...e.dataTransfer.files]);
});

// ------------------------------------------------------------------ 진행률(SSE)
function connectEvents() {
  const es = new EventSource("/api/events");
  es.onmessage = async (e) => {
    const msg = JSON.parse(e.data);
    if (msg.type === "progress") {
      const box = $("#progress");
      box.hidden = false;
      const pct = msg.total ? (msg.done / msg.total) * 100 : 0;
      box.querySelector(".fill").style.width = pct + "%";
      box.querySelector(".label").textContent =
        msg.done >= msg.total ? "완료" : `${msg.done}/${msg.total} — ${msg.current}`;
    } else if (msg.type === "done") {
      setTimeout(() => { $("#progress").hidden = true; }, 1200);
      applySnapshot(await api("/api/state"));
    } else if (msg.type === "model") {
      onModelEvent(msg);
    } else if (msg.type === "error") {
      toast(`${msg.filename}: ${msg.message}`, "bad");
    }
  };
  es.onerror = () => { es.close(); setTimeout(connectEvents, 3000); };
}

// ------------------------------------------------------------------ 검출 모델
function modelById(id) {
  return state.models.find((m) => m.id === id);
}

/** 편집 창의 드롭다운을 현재 모델 목록으로 다시 그린다. */
function syncDetectorSelect(current) {
  const sel = $("#detectorSelect");
  if (!sel) return;
  sel.innerHTML = "";
  const list = state.models.length ? state.models
    : [{ id: current, name: current, ready: true }];
  for (const m of list) {
    const o = document.createElement("option");
    o.value = m.id;
    o.textContent = m.ready ? m.name : `${m.name} — 설치 필요`;
    sel.appendChild(o);
  }
  sel.value = current;
  const m = modelById(current);
  $("#detectorHint").textContent = m
    ? (m.ready ? m.summary : `${m.summary} (위 '검출 모델'에서 설치하세요)`)
    : "";
}

$("#detectorSelect").onchange = (e) => {
  if (!state.editing) return;
  state.editing.settings.detector = e.target.value;
  syncControls();
};

async function loadModels() {
  const r = await api("/api/models");
  state.models = r.models;
  state.selectedModel = r.selected;
  return r;
}

function renderModelList() {
  const box = $("#modelList");
  box.innerHTML = "";
  for (const m of state.models) {
    const row = document.createElement("label");
    row.className = "modelrow" + (m.ready ? "" : " missing");

    const radio = document.createElement("input");
    radio.type = "radio";
    radio.name = "model";
    radio.value = m.id;
    radio.checked = m.id === state.selectedModel;
    radio.onchange = () => { state.selectedModel = m.id; };

    const body = document.createElement("div");
    const title = document.createElement("div");
    title.className = "modelname";
    title.textContent = m.name;
    if (!m.ready) {
      const badge = document.createElement("span");
      badge.className = "badge warn";
      badge.textContent = m.missing.join(" + ") + " 없음";
      title.appendChild(badge);
    }
    const desc = document.createElement("p");
    desc.className = "hint small";
    desc.textContent = m.summary;
    body.append(title, desc);

    if (m.source) {
      const src = document.createElement("p");
      src.className = "hint small";
      src.textContent = `출처 ${m.source} — ${m.license}`;
      body.appendChild(src);
    }
    if (!m.ready) {
      const btn = document.createElement("button");
      btn.className = "ghost";
      const mb = Math.round(m.weights_bytes / 1e6);
      btn.textContent = `설치 (torch + 가중치 ${mb}MB 내려받기)`;
      btn.onclick = (e) => { e.preventDefault(); installModel(m.id, btn); };
      body.appendChild(btn);
    }
    row.append(radio, body);
    box.appendChild(row);
  }
}

let installingBtn = null;
async function installModel(id, btn) {
  btn.disabled = true;
  btn.textContent = "설치 중…";
  installingBtn = btn;
  const box = $("#modelProgress");
  box.hidden = false;
  box.querySelector(".fill").style.width = "0%";
  box.querySelector(".label").textContent = "시작하는 중…";
  try {
    await api(`/api/models/${id}/install`, { method: "POST" });
  } catch (err) {
    toast("설치를 시작하지 못했습니다: " + err.message, "bad");
    box.hidden = true;
    btn.disabled = false;
    btn.textContent = "설치";
    installingBtn = null;
  }
}

async function onModelEvent(msg) {
  const box = $("#modelProgress");
  const fill = box.querySelector(".fill");
  const label = box.querySelector(".label");
  box.hidden = false;
  if (msg.phase === "packages") {
    fill.style.width = "8%";
    label.textContent = msg.message || "torch 설치 중… (수백 MB, 몇 분 걸립니다)";
  } else if (msg.phase === "weights") {
    // 다운로드 구간을 10~100%에 매핑한다. 앞의 10%는 패키지 설치 몫.
    const pct = msg.total ? 10 + (msg.done / msg.total) * 90 : 10;
    fill.style.width = pct + "%";
    label.textContent = `가중치 내려받는 중 ${(msg.done / 1e6).toFixed(0)}`
      + `/${(msg.total / 1e6).toFixed(0)} MB`;
  } else if (msg.phase === "done") {
    fill.style.width = "100%";
    label.textContent = "설치 완료";
    await loadModels();
    renderModelList();
    setTimeout(() => { box.hidden = true; }, 1500);
    toast("모델 설치 완료", "good");
    installingBtn = null;
  } else if (msg.phase === "error") {
    box.hidden = true;
    toast("설치 실패: " + msg.message, "bad");
    if (installingBtn) { installingBtn.disabled = false; installingBtn.textContent = "설치 다시 시도"; }
    installingBtn = null;
  }
}

$("#modelBtn").onclick = async () => {
  $("#modelPanel").hidden = false;
  try {
    await loadModels();
    renderModelList();
  } catch (err) {
    toast("모델 목록을 불러오지 못했습니다: " + err.message, "bad");
  }
};

$("#applyModel").onclick = async () => {
  const btn = $("#applyModel");
  const chosen = modelById(state.selectedModel);
  if (chosen && !chosen.ready) {
    toast("먼저 설치해야 이 모델을 쓸 수 있습니다.", "bad");
    return;
  }
  btn.disabled = true;
  try {
    const r = await api("/api/models/select", {
      method: "POST",
      body: JSON.stringify({ id: state.selectedModel, apply_all: $("#modelApplyAll").checked }),
    });
    $("#modelPanel").hidden = true;
    toast(r.reprocessed ? `${r.reprocessed}장을 다시 처리합니다` : "모델을 바꿨습니다", "good");
  } catch (err) {
    toast("적용 실패: " + err.message, "bad");
  } finally { btn.disabled = false; }
};

// ------------------------------------------------------------------ 카메라
$("#cameraBtn").onclick = async () => {
  $("#cameraPanel").hidden = false;
  try {
    const r = await api("/api/cameras");
    const sel = $("#deviceSelect");
    sel.innerHTML = "";
    if (!r.devices.length) {
      sel.innerHTML = `<option>카메라를 찾지 못했습니다</option>`;
      $("#openCamera").disabled = true;
    } else {
      $("#openCamera").disabled = false;
      r.devices.forEach((d) => {
        const o = document.createElement("option");
        o.value = d.index; o.textContent = `${d.index}: ${d.name}`;
        sel.appendChild(o);
      });
      if (r.active != null) sel.value = r.active;
    }
    $("#cameraHint").textContent = r.hint;
  } catch (err) { toast("카메라 목록 실패: " + err.message, "bad"); }
};

$("#openCamera").onclick = async () => {
  const idx = parseInt($("#deviceSelect").value, 10);
  const btn = $("#openCamera");
  btn.disabled = true; btn.textContent = "연결 중…";
  try {
    await api("/api/camera/open", { method: "POST", body: JSON.stringify({ index: idx }) });
    state.cameraOn = true;
    const view = $("#cameraView");
    view.src = `/api/camera/stream?t=${Date.now()}`;
    view.classList.add("on");
    $("#cameraPlaceholder").style.display = "none";
    toast("카메라 연결됨", "good");
  } catch (err) { toast("연결 실패: " + err.message, "bad"); }
  finally { btn.disabled = false; btn.textContent = "연결"; }
};

async function closeCamera() {
  const view = $("#cameraView");
  view.src = ""; view.classList.remove("on");
  $("#cameraPlaceholder").style.display = "";
  state.cameraOn = false;
  try { await api("/api/camera/close", { method: "POST" }); } catch { /* 이미 닫힘 */ }
}
$("#closeCamera").onclick = closeCamera;

let captured = 0;
$("#captureBtn").onclick = async () => {
  if (!state.cameraOn) { toast("먼저 카메라를 연결하세요.", "bad"); return; }
  const btn = $("#captureBtn");
  btn.disabled = true;
  try {
    applySnapshot(await api("/api/camera/capture", { method: "POST" }));
    captured++;
    $("#captureCount").textContent = `${captured}장 촬영됨`;
    toast("촬영 완료", "good");
  } catch (err) { toast("촬영 실패: " + err.message, "bad"); }
  finally { btn.disabled = false; }
};

// ------------------------------------------------------------------ PDF
$("#pdfBtn").onclick = () => {
  $("#pdfPanel").hidden = false;
  $("#pdfName").value = state.project;
  $("#pdfInfo").textContent = `${state.pages.length}장이 현재 순서대로 들어갑니다.`;
};

$("#makePdf").onclick = async () => {
  const btn = $("#makePdf");
  btn.disabled = true; btn.textContent = "만드는 중…";
  try {
    const r = await api("/api/pdf", {
      method: "POST",
      body: JSON.stringify({ filename: $("#pdfName").value.trim(), size: $("#pdfSize").value }),
    });
    $("#pdfInfo").textContent = `${r.name} — ${r.pages}장, ${(r.size / 1048576).toFixed(1)} MB`;
    toast(`PDF 생성 완료: ${r.name}`, "good");
    await api("/api/reveal", { method: "POST", body: JSON.stringify({ path: r.path }) });
  } catch (err) { toast("PDF 실패: " + err.message, "bad"); }
  finally { btn.disabled = false; btn.textContent = "만들기"; }
};

$("#revealBtn").onclick = async () => {
  try { await api("/api/reveal", { method: "POST", body: JSON.stringify({}) }); }
  catch (err) { toast("폴더 열기 실패: " + err.message, "bad"); }
};

// ------------------------------------------------------------------ 프로젝트
$("#projectSelect").onchange = async (e) => {
  try { applySnapshot(await api("/api/projects", { method: "POST", body: JSON.stringify({ name: e.target.value }) })); }
  catch (err) { toast("프로젝트 전환 실패: " + err.message, "bad"); }
};

$("#newProject").onclick = async () => {
  const name = prompt("새 프로젝트 이름");
  if (!name) return;
  try {
    const data = await api("/api/projects", { method: "POST", body: JSON.stringify({ name }) });
    applySnapshot(data);
    fillProjects(data.projects, data.project);
  } catch (err) { toast("생성 실패: " + err.message, "bad"); }
};

function fillProjects(names, current) {
  const sel = $("#projectSelect");
  sel.innerHTML = "";
  (names || []).forEach((n) => {
    const o = document.createElement("option");
    o.value = n; o.textContent = n;
    sel.appendChild(o);
  });
  if (current) sel.value = current;
}

// ------------------------------------------------------------------ 공통
document.querySelectorAll("[data-close]").forEach((b) => b.onclick = () => {
  const overlay = b.closest(".overlay");
  overlay.hidden = true;
  if (overlay.id === "cameraPanel") closeCamera();
});

document.querySelectorAll(".overlay").forEach((ov) => ov.addEventListener("mousedown", (e) => {
  if (e.target === ov) {
    ov.hidden = true;
    if (ov.id === "cameraPanel") closeCamera();
  }
}));

window.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    document.querySelectorAll(".overlay:not([hidden])").forEach((ov) => {
      ov.hidden = true;
      if (ov.id === "cameraPanel") closeCamera();
    });
  }
  // 카메라 패널이 열려 있을 때 스페이스바로 촬영
  if (e.code === "Space" && !$("#cameraPanel").hidden && e.target.tagName !== "INPUT") {
    e.preventDefault();
    $("#captureBtn").click();
  }
});

window.addEventListener("resize", () => {
  if (state.editing && editImg && !$("#editor").hidden) { fitCanvas(); drawCanvas(); }
});

(async function init() {
  try {
    const data = await api("/api/state");
    applySnapshot(data);
    fillProjects(data.projects, data.project);
    $("#folderInput").value = data.default_input || "";
    connectEvents();
    loadModels().catch(() => { /* 모델 목록은 없어도 편집은 된다 */ });
  } catch (err) {
    toast("서버에 연결할 수 없습니다: " + err.message, "bad");
  }
})();
