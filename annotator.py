from pathlib import Path
from flask import Flask, jsonify, request, send_file, abort

IMAGES_FOLDER    = Path(r"D:\LeavesDatasetClassic\test")
LABELS_FOLDER    = Path(r"D:\LeavesDatasetClassic\predictions\labels")   # predicted labels (read)
CORRECTED_FOLDER = Path(r"D:\LeavesDatasetClassic\corrected\labels")     # corrected labels (write)

CLASS_NAMES = {0: "fruits", 1: "leaves"}   # must match dataset.yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
app = Flask(__name__)


def image_list():
    return sorted(p.name for p in IMAGES_FOLDER.iterdir() if p.suffix.lower() in IMG_EXTS)

def read_labels(img_name: str):
    # prefer already-corrected version if it exists
    for folder in (CORRECTED_FOLDER, LABELS_FOLDER):
        lp = folder / (Path(img_name).stem + ".txt")
        if lp.exists():
            boxes = []
            for line in lp.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) == 5:
                    boxes.append({
                        "cls": int(parts[0]),
                        "cx":  float(parts[1]),
                        "cy":  float(parts[2]),
                        "w":   float(parts[3]),
                        "h":   float(parts[4]),
                    })
            return boxes
    return []

def write_labels(img_name: str, boxes):
    CORRECTED_FOLDER.mkdir(parents=True, exist_ok=True)
    out = CORRECTED_FOLDER / (Path(img_name).stem + ".txt")
    lines = [
        f"{b['cls']} {b['cx']:.6f} {b['cy']:.6f} {b['w']:.6f} {b['h']:.6f}"
        for b in boxes
    ]
    out.write_text("\n".join(lines))


@app.route("/api/images")
def api_images():
    imgs = image_list()
    corrected = {p.stem for p in CORRECTED_FOLDER.glob("*.txt")} if CORRECTED_FOLDER.exists() else set()
    return jsonify([{"name": n, "corrected": Path(n).stem in corrected} for n in imgs])

@app.route("/api/image/<path:name>")
def api_image(name):
    p = IMAGES_FOLDER / name
    if not p.exists():
        abort(404)
    return send_file(str(p))

@app.route("/api/labels/<path:name>", methods=["GET"])
def api_labels_get(name):
    return jsonify(read_labels(name))

@app.route("/api/labels/<path:name>", methods=["POST"])
def api_labels_post(name):
    data = request.get_json()
    write_labels(name, data.get("boxes", []))
    return jsonify({"ok": True})

@app.route("/api/classes")
def api_classes():
    return jsonify(CLASS_NAMES)

# ── HTML ──────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Leaf Annotator</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f0f13; color: #e0e0e8; height: 100vh; display: flex; flex-direction: column; overflow: hidden; }

  /* top bar */
  #topbar { display: flex; align-items: center; gap: 10px; padding: 7px 14px; background: #1a1a24; border-bottom: 1px solid #2e2e42; flex-shrink: 0; flex-wrap: wrap; }
  #topbar h1 { font-size: .95rem; font-weight: 700; color: #a78bfa; white-space: nowrap; }
  .sep { width: 1px; height: 18px; background: #2e2e42; flex-shrink: 0; }
  #img-counter { font-size: .8rem; color: #888; white-space: nowrap; }
  #status-msg { font-size: .8rem; color: #6ee7b7; min-width: 100px; }

  /* zoom controls */
  #zoom-controls { display: flex; align-items: center; gap: 4px; }
  .zoom-btn { padding: 3px 9px; border: 1px solid #3e3e5e; border-radius: 4px; background: #1e1e2e; color: #aaa; font-size: .85rem; cursor: pointer; line-height: 1; }
  .zoom-btn:hover { background: #2e2e48; color: #fff; }
  #zoom-label { font-size: .78rem; color: #666; min-width: 42px; text-align: center; }

  /* label toggle */
  #label-toggle { padding: 3px 10px; border: 1px solid #3e3e5e; border-radius: 4px; background: #1e1e2e; color: #aaa; font-size: .78rem; cursor: pointer; white-space: nowrap; }
  #label-toggle.on { background: #3e2e5e; color: #c4b5fd; border-color: #7c3aed; }

  #save-btn { margin-left: auto; padding: 5px 16px; border: none; border-radius: 6px; background: #7c3aed; color: #fff; font-size: .82rem; font-weight: 700; cursor: pointer; white-space: nowrap; }
  #save-btn:hover { background: #6d28d9; }
  #save-btn.saved { background: #16a34a; }

  /* main layout */
  #main { display: flex; flex: 1; overflow: hidden; }

  /* left sidebar */
  #sidebar-left { width: 190px; flex-shrink: 0; background: #13131c; border-right: 1px solid #2e2e42; display: flex; flex-direction: column; overflow: hidden; }
  #sidebar-left h2 { font-size: .7rem; text-transform: uppercase; letter-spacing: 1px; color: #555; padding: 9px 12px 5px; flex-shrink: 0; }
  #image-list { flex: 1; overflow-y: auto; }
  .img-item { padding: 6px 10px; font-size: .75rem; cursor: pointer; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; border-left: 3px solid transparent; }
  .img-item:hover { background: #1e1e2e; }
  .img-item.active { background: #1e1e2e; border-left-color: #7c3aed; color: #c4b5fd; }
  .img-item .badge { float: right; font-size: .65rem; padding: 1px 4px; border-radius: 3px; background: #16a34a22; color: #4ade80; }

  /* canvas */
  #canvas-wrap { flex: 1; overflow: hidden; background: #181820; position: relative; }
  #main-canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; cursor: crosshair; display: block; }
  #loading-overlay { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; background: #181820; font-size: 1rem; color: #555; pointer-events: none; }

  /* right sidebar */
  #sidebar-right { width: 210px; flex-shrink: 0; background: #13131c; border-left: 1px solid #2e2e42; display: flex; flex-direction: column; overflow: hidden; }
  .panel-section { padding: 10px 13px; border-bottom: 1px solid #1e1e2e; flex-shrink: 0; }
  .panel-section label { display: block; font-size: .72rem; color: #777; margin-bottom: 5px; }
  #class-select { width: 100%; padding: 5px 7px; background: #1e1e2e; border: 1px solid #3e3e5e; border-radius: 5px; color: #e0e0e8; font-size: .82rem; }
  #del-btn { width: 100%; padding: 7px; border: none; border-radius: 5px; background: #7f1d1d; color: #fca5a5; font-size: .82rem; font-weight: 600; cursor: pointer; }
  #del-btn:hover:not(:disabled) { background: #991b1b; }
  #del-btn:disabled { opacity: .3; cursor: default; }

  .legend-row { display: flex; align-items: center; gap: 8px; margin: 3px 0; font-size: .78rem; }
  .legend-dot { width: 12px; height: 12px; border-radius: 3px; flex-shrink: 0; }

  #nav-btns { display: flex; gap: 6px; padding: 9px 13px; border-bottom: 1px solid #1e1e2e; flex-shrink: 0; }
  #nav-btns button { flex: 1; padding: 6px; border: none; border-radius: 5px; background: #1e1e2e; color: #aaa; font-size: .95rem; cursor: pointer; }
  #nav-btns button:hover { background: #2e2e48; color: #fff; }

  #box-list-wrap { flex: 1; overflow-y: auto; }
  #box-list-header { font-size: .7rem; text-transform: uppercase; letter-spacing: 1px; color: #555; padding: 8px 13px 4px; }
  #box-list { padding: 0 8px 8px; }
  .box-row { padding: 5px 6px; border-radius: 4px; font-size: .74rem; cursor: pointer; display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px; border: 1px solid transparent; }
  .box-row:hover { background: #1e1e2e; }
  .box-row.selected { background: #1e1e38; border-color: #7c3aed; }
  .box-cls-tag { padding: 1px 5px; border-radius: 3px; font-size: .68rem; font-weight: 700; }

  /* save path */
  #save-path { font-size: .68rem; color: #444; padding: 8px 13px; line-height: 1.5; border-top: 1px solid #1e1e2e; flex-shrink: 0; word-break: break-all; }
  #save-path span { color: #666; }

  /* hints */
  #hints { padding: 9px 13px; font-size: .7rem; color: #444; line-height: 1.9; border-top: 1px solid #1e1e2e; flex-shrink: 0; }
  kbd { background: #1e1e2e; border: 1px solid #3e3e5e; border-radius: 3px; padding: 0 3px; font-size: .68rem; color: #888; }
</style>
</head>
<body>

<div id="topbar">
  <h1>🍃 Leaf Annotator</h1>
  <div class="sep"></div>
  <span id="img-counter">–</span>
  <div class="sep"></div>

  <!-- zoom -->
  <div id="zoom-controls">
    <button class="zoom-btn" id="zoom-out">−</button>
    <span id="zoom-label">100%</span>
    <button class="zoom-btn" id="zoom-in">+</button>
    <button class="zoom-btn" id="zoom-fit">Fit</button>
  </div>

  <div class="sep"></div>
  <button id="label-toggle" class="on">Labels ON</button>
  <div class="sep"></div>
  <span id="status-msg"></span>
  <button id="save-btn">Save  Ctrl+S</button>
</div>

<div id="main">
  <!-- image list -->
  <div id="sidebar-left">
    <h2>Images</h2>
    <div id="image-list"></div>
  </div>

  <!-- canvas -->
  <div id="canvas-wrap">
    <canvas id="main-canvas"></canvas>
    <div id="loading-overlay">Loading…</div>
  </div>

  <!-- right panel -->
  <div id="sidebar-right">
    <div class="panel-section">
      <label>Class (new / selected box)</label>
      <select id="class-select"></select>
    </div>
    <div class="panel-section">
      <button id="del-btn" disabled>Delete selected  Del</button>
    </div>
    <div class="panel-section">
      <label>Legend</label>
      <div id="legend"></div>
    </div>
    <div id="nav-btns">
      <button id="prev-btn">◀ Prev</button>
      <button id="next-btn">Next ▶</button>
    </div>

    <div id="box-list-wrap">
      <div id="box-list-header">Boxes (<span id="box-count">0</span>)</div>
      <div id="box-list"></div>
    </div>

    <div id="save-path"><span>Saving to:</span><br>CORRECTED_FOLDER_PLACEHOLDER</div>

    <div id="hints">
      <kbd>Wheel</kbd> zoom &nbsp; <kbd>RMB drag</kbd> pan<br>
      <kbd>←</kbd><kbd>→</kbd> navigate<br>
      <kbd>Del</kbd> delete &nbsp; <kbd>Esc</kbd> deselect<br>
      <kbd>Ctrl+S</kbd> save<br>
      Drag empty → new box<br>
      Drag box → move<br>
      Drag corner → resize
    </div>
  </div>
</div>

<script>
// ── class colours ─────────────────────────────────────────────────────────────
const COLORS = { 0: '#f97316', 1: '#a855f7' };
function clsColor(cls) { return COLORS[cls] ?? '#22c55e'; }

// ── state ─────────────────────────────────────────────────────────────────────
let images   = [];
let imgIdx   = 0;
let boxes    = [];
let selIdx   = -1;
let classMap = {};
let showLabels = true;

let imgEl  = null;
let baseW  = 1, baseH = 1;   // image drawn at this size at zoom=1
let zoom   = 1;
let panX   = 0, panY = 0;

const canvas  = document.getElementById('main-canvas');
const ctx     = canvas.getContext('2d');
const wrap    = document.getElementById('canvas-wrap');

// drag state
let drag = null;
// {type:'move', boxIdx, startMx, startMy, origCx, origCy}
// {type:'resize', boxIdx, handle, startMx, startMy, origBox}
// {type:'new', startNx, startNy, curNx, curNy}
// {type:'pan', startMx, startMy, origPanX, origPanY}

const HANDLE_R = 6;   // px in image space

// ── coordinate helpers ────────────────────────────────────────────────────────
// mouse event → canvas pixel
function canvasPt(e) {
  const r = canvas.getBoundingClientRect();
  const scaleX = canvas.width  / r.width;
  const scaleY = canvas.height / r.height;
  return { mx: (e.clientX - r.left) * scaleX, my: (e.clientY - r.top) * scaleY };
}
// canvas pixel → image space pixel
function toImg(mx, my) {
  return { ix: (mx - panX) / zoom, iy: (my - panY) / zoom };
}
// image space pixel → canvas pixel
function toScr(ix, iy) {
  return { sx: ix * zoom + panX, sy: iy * zoom + panY };
}
// box normalised → image-space rect
function boxImg(b) {
  return {
    x1: (b.cx - b.w/2) * baseW,
    y1: (b.cy - b.h/2) * baseH,
    x2: (b.cx + b.w/2) * baseW,
    y2: (b.cy + b.h/2) * baseH,
  };
}
// mouse → normalised
function toNorm(mx, my) {
  const { ix, iy } = toImg(mx, my);
  return { nx: ix / baseW, ny: iy / baseH };
}
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

function hitHandle(b, ix, iy) {
  const { x1, y1, x2, y2 } = boxImg(b);
  const pts = { tl:[x1,y1], tr:[x2,y1], bl:[x1,y2], br:[x2,y2] };
  const r = HANDLE_R / zoom;   // handle radius in image space grows as you zoom out
  for (const [name, [hx, hy]] of Object.entries(pts)) {
    if (Math.hypot(ix - hx, iy - hy) <= r + 2) return name;
  }
  return null;
}
function hitBox(b, ix, iy) {
  const { x1, y1, x2, y2 } = boxImg(b);
  return ix >= x1 && ix <= x2 && iy >= y1 && iy <= y2;
}

// ── zoom / pan ────────────────────────────────────────────────────────────────
const MIN_ZOOM = 0.1, MAX_ZOOM = 10;

function zoomAt(factor, cx, cy) {
  const newZoom = clamp(zoom * factor, MIN_ZOOM, MAX_ZOOM);
  // keep point (cx,cy) fixed on screen
  panX = cx - (cx - panX) * (newZoom / zoom);
  panY = cy - (cy - panY) * (newZoom / zoom);
  zoom = newZoom;
  updateZoomLabel();
  redraw();
}
function fitZoom() {
  if (!imgEl) return;
  zoom = Math.min(canvas.width / baseW, canvas.height / baseH);
  panX = (canvas.width  - baseW * zoom) / 2;
  panY = (canvas.height - baseH * zoom) / 2;
  updateZoomLabel();
}
function updateZoomLabel() {
  document.getElementById('zoom-label').textContent = Math.round(zoom * 100) + '%';
}

// ── draw ──────────────────────────────────────────────────────────────────────
function redraw() {
  if (!imgEl) return;
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  ctx.save();
  ctx.translate(panX, panY);
  ctx.scale(zoom, zoom);

  // image
  ctx.drawImage(imgEl, 0, 0, baseW, baseH);

  // boxes
  const fontSize = Math.max(10, Math.min(18, 13 / zoom));
  ctx.font = `${fontSize}px 'Segoe UI', sans-serif`;

  boxes.forEach((b, i) => {
    const { x1, y1, x2, y2 } = boxImg(b);
    const color  = clsColor(b.cls);
    const isSel  = i === selIdx;
    const lw     = (isSel ? 3 : 2) / zoom;

    ctx.strokeStyle = color;
    ctx.lineWidth   = lw;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    if (isSel) {
      ctx.fillStyle = color + '28';
      ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
    }

    // class label (toggleable)
    if (showLabels) {
      const label = classMap[b.cls] ?? String(b.cls);
      const tw    = ctx.measureText(label).width;
      const th    = fontSize;
      const tx    = x1;
      const ty    = y1 > th + 4 / zoom ? y1 - 2 / zoom : y2 + th + 2 / zoom;
      const pad   = 3 / zoom;
      ctx.fillStyle = color;
      ctx.fillRect(tx - pad, ty - th - pad, tw + pad * 4, th + pad * 2);
      ctx.fillStyle = '#fff';
      ctx.fillText(label, tx + pad, ty - pad / 2);
    }

    // resize handles (selected only)
    if (isSel) {
      const hr = (HANDLE_R + 1) / zoom;
      [[x1,y1],[x2,y1],[x1,y2],[x2,y2]].forEach(([hx, hy]) => {
        ctx.fillStyle = '#fff';
        ctx.beginPath();
        ctx.arc(hx, hy, hr, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5 / zoom;
        ctx.stroke();
      });
    }
  });

  ctx.restore();
}

// ── canvas events ─────────────────────────────────────────────────────────────
canvas.addEventListener('wheel', e => {
  e.preventDefault();
  const { mx, my } = canvasPt(e);
  zoomAt(e.deltaY < 0 ? 1.12 : 1 / 1.12, mx, my);
}, { passive: false });

canvas.addEventListener('mousedown', e => {
  const { mx, my } = canvasPt(e);

  // right-click → pan
  if (e.button === 1 || e.button === 2) {
    drag = { type: 'pan', startMx: mx, startMy: my, origPanX: panX, origPanY: panY };
    canvas.style.cursor = 'grabbing';
    return;
  }

  if (e.button !== 0) return;
  const { ix, iy } = toImg(mx, my);

  // check resize handle on selected box
  if (selIdx >= 0) {
    const h = hitHandle(boxes[selIdx], ix, iy);
    if (h) {
      drag = { type: 'resize', boxIdx: selIdx, handle: h,
               startMx: mx, startMy: my, origBox: { ...boxes[selIdx] } };
      return;
    }
  }

  // check hit on any box (top-most first)
  for (let i = boxes.length - 1; i >= 0; i--) {
    if (hitBox(boxes[i], ix, iy)) {
      selIdx = i;
      drag = { type: 'move', boxIdx: i, startMx: mx, startMy: my,
               origCx: boxes[i].cx, origCy: boxes[i].cy };
      updateUI(); redraw();
      return;
    }
  }

  // start new box
  selIdx = -1;
  const { nx, ny } = toNorm(mx, my);
  drag = { type: 'new', startNx: nx, startNy: ny, curNx: nx, curNy: ny };
  updateUI(); redraw();
});

canvas.addEventListener('mousemove', e => {
  const { mx, my } = canvasPt(e);

  if (!drag) {
    const { ix, iy } = toImg(mx, my);
    let cur = 'crosshair';
    if (selIdx >= 0 && hitHandle(boxes[selIdx], ix, iy)) cur = 'nwse-resize';
    else if (boxes.some(b => hitBox(b, ix, iy))) cur = 'move';
    canvas.style.cursor = cur;
    return;
  }

  if (drag.type === 'pan') {
    panX = drag.origPanX + (mx - drag.startMx);
    panY = drag.origPanY + (my - drag.startMy);
    redraw();
    return;
  }

  if (drag.type === 'move') {
    const b  = boxes[drag.boxIdx];
    const dx = (mx - drag.startMx) / zoom / baseW;
    const dy = (my - drag.startMy) / zoom / baseH;
    b.cx = clamp(drag.origCx + dx, b.w/2, 1 - b.w/2);
    b.cy = clamp(drag.origCy + dy, b.h/2, 1 - b.h/2);
    redraw();
    return;
  }

  if (drag.type === 'resize') {
    const b  = boxes[drag.boxIdx];
    const ob = drag.origBox;
    const dx = (mx - drag.startMx) / zoom / baseW;
    const dy = (my - drag.startMy) / zoom / baseH;
    let { x1, y1, x2, y2 } = {
      x1: ob.cx - ob.w/2, y1: ob.cy - ob.h/2,
      x2: ob.cx + ob.w/2, y2: ob.cy + ob.h/2,
    };
    const h = drag.handle;
    if (h.includes('l')) x1 = clamp(x1 + dx, 0, x2 - 0.005);
    if (h.includes('r')) x2 = clamp(x2 + dx, x1 + 0.005, 1);
    if (h.includes('t')) y1 = clamp(y1 + dy, 0, y2 - 0.005);
    if (h.includes('b')) y2 = clamp(y2 + dy, y1 + 0.005, 1);
    b.cx = (x1+x2)/2; b.cy = (y1+y2)/2; b.w = x2-x1; b.h = y2-y1;
    redraw();
    return;
  }

  if (drag.type === 'new') {
    const { nx, ny } = toNorm(mx, my);
    drag.curNx = clamp(nx, 0, 1);
    drag.curNy = clamp(ny, 0, 1);
    redraw();
    // dashed preview
    ctx.save();
    ctx.translate(panX, panY);
    ctx.scale(zoom, zoom);
    const x1 = Math.min(drag.startNx, drag.curNx) * baseW;
    const y1 = Math.min(drag.startNy, drag.curNy) * baseH;
    const x2 = Math.max(drag.startNx, drag.curNx) * baseW;
    const y2 = Math.max(drag.startNy, drag.curNy) * baseH;
    ctx.strokeStyle = clsColor(selectedClass());
    ctx.lineWidth   = 2 / zoom;
    ctx.setLineDash([6 / zoom, 3 / zoom]);
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.setLineDash([]);
    ctx.restore();
  }
});

canvas.addEventListener('mouseup',    finishDrag);
canvas.addEventListener('mouseleave', finishDrag);
canvas.addEventListener('contextmenu', e => e.preventDefault());

function finishDrag(e) {
  if (!drag) return;

  if (drag.type === 'new') {
    const { nx, ny } = (() => {
      if (e.type === 'mouseleave') return { nx: drag.curNx, ny: drag.curNy };
      const { mx, my } = canvasPt(e);
      return toNorm(mx, my);
    })();
    const cx = (drag.startNx + nx) / 2;
    const cy = (drag.startNy + ny) / 2;
    const bw = Math.abs(nx - drag.startNx);
    const bh = Math.abs(ny - drag.startNy);
    if (bw > 0.004 && bh > 0.004) {
      boxes.push({ cls: selectedClass(), cx, cy, w: bw, h: bh });
      selIdx = boxes.length - 1;
    }
  }

  drag = null;
  canvas.style.cursor = 'crosshair';
  updateUI(); redraw(); renderBoxList();
}

// ── load image ────────────────────────────────────────────────────────────────
async function loadImage(idx) {
  if (idx < 0 || idx >= images.length) return;
  imgIdx = idx;
  selIdx = -1; drag = null;
  document.getElementById('loading-overlay').style.display = 'flex';

  const name = images[imgIdx].name;
  document.getElementById('img-counter').textContent =
    `${imgIdx + 1} / ${images.length}  —  ${name}`;

  boxes = await fetch(`/api/labels/${encodeURIComponent(name)}`).then(r => r.json());

  imgEl = new Image();
  imgEl.onload = () => {
    // fit canvas to wrap
    canvas.width  = wrap.clientWidth;
    canvas.height = wrap.clientHeight;
    // base size: fit image keeping aspect ratio
    const nat = imgEl.naturalWidth / imgEl.naturalHeight;
    const wrp = canvas.width / canvas.height;
    if (nat > wrp) {
      baseW = canvas.width * 0.95;
      baseH = baseW / nat;
    } else {
      baseH = canvas.height * 0.95;
      baseW = baseH * nat;
    }
    fitZoom();
    redraw();
    document.getElementById('loading-overlay').style.display = 'none';
    updateUI(); renderBoxList();
  };
  imgEl.src = `/api/image/${encodeURIComponent(name)}`;

  document.querySelectorAll('.img-item').forEach((el, i) => {
    el.classList.toggle('active', i === idx);
  });
  document.querySelector(`.img-item[data-idx="${idx}"]`)?.scrollIntoView({ block: 'nearest' });
  setStatus('');
  document.getElementById('save-btn').classList.remove('saved');
}

// ── init ──────────────────────────────────────────────────────────────────────
async function init() {
  const [imgs, clsMap] = await Promise.all([
    fetch('/api/images').then(r => r.json()),
    fetch('/api/classes').then(r => r.json()),
  ]);
  images   = imgs;
  classMap = clsMap;

  // class selector
  const sel = document.getElementById('class-select');
  for (const [id, name] of Object.entries(classMap)) {
    const opt = document.createElement('option');
    opt.value = id; opt.textContent = name;
    sel.appendChild(opt);
  }

  // legend
  const leg = document.getElementById('legend');
  for (const [id, name] of Object.entries(classMap)) {
    const row = document.createElement('div');
    row.className = 'legend-row';
    row.innerHTML = `<div class="legend-dot" style="background:${clsColor(Number(id))}"></div><span>${name}</span>`;
    leg.appendChild(row);
  }

  renderImageList();
  loadImage(0);

  // buttons
  document.getElementById('save-btn').addEventListener('click', save);
  document.getElementById('del-btn').addEventListener('click', deleteSelected);
  document.getElementById('prev-btn').addEventListener('click', () => navigate(-1));
  document.getElementById('next-btn').addEventListener('click', () => navigate(1));
  document.getElementById('zoom-in').addEventListener('click', () => {
    const c = canvas.width/2, r = canvas.height/2; zoomAt(1.25, c, r);
  });
  document.getElementById('zoom-out').addEventListener('click', () => {
    const c = canvas.width/2, r = canvas.height/2; zoomAt(1/1.25, c, r);
  });
  document.getElementById('zoom-fit').addEventListener('click', () => { fitZoom(); redraw(); });
  document.getElementById('label-toggle').addEventListener('click', e => {
    showLabels = !showLabels;
    e.target.textContent = showLabels ? 'Labels ON' : 'Labels OFF';
    e.target.classList.toggle('on', showLabels);
    redraw();
  });
  document.getElementById('class-select').addEventListener('change', e => {
    if (selIdx >= 0) { boxes[selIdx].cls = Number(e.target.value); redraw(); renderBoxList(); }
  });

  // keyboard
  document.addEventListener('keydown', e => {
    if (e.target.tagName === 'SELECT') return;
    if (e.key === 'Delete' || e.key === 'Backspace') deleteSelected();
    if (e.key === 'Escape') { selIdx = -1; updateUI(); redraw(); renderBoxList(); }
    if (e.key === 'ArrowRight') navigate(1);
    if (e.key === 'ArrowLeft')  navigate(-1);
    if (e.key === 's' && e.ctrlKey) { e.preventDefault(); save(); }
  });

  // resize
  window.addEventListener('resize', () => {
    if (!imgEl) return;
    canvas.width  = wrap.clientWidth;
    canvas.height = wrap.clientHeight;
    fitZoom(); redraw();
  });
}

// ── helpers ───────────────────────────────────────────────────────────────────
function renderImageList() {
  const list = document.getElementById('image-list');
  list.innerHTML = '';
  images.forEach((img, i) => {
    const el = document.createElement('div');
    el.className = 'img-item';
    el.dataset.idx = i;
    el.innerHTML = `${img.name}${img.corrected ? '<span class="badge">✓</span>' : ''}`;
    el.addEventListener('click', () => loadImage(i));
    list.appendChild(el);
  });
}

function renderBoxList() {
  document.getElementById('box-count').textContent = boxes.length;
  const list = document.getElementById('box-list');
  list.innerHTML = '';
  boxes.forEach((b, i) => {
    const el = document.createElement('div');
    el.className = 'box-row' + (i === selIdx ? ' selected' : '');
    const name  = classMap[b.cls] ?? String(b.cls);
    const color = clsColor(b.cls);
    el.innerHTML = `<span class="box-cls-tag" style="background:${color}22;color:${color}">${name}</span>
                    <span style="color:#444">#${i+1}</span>`;
    el.addEventListener('click', () => { selIdx = i; updateUI(); redraw(); renderBoxList(); });
    list.appendChild(el);
  });
}

function updateUI() {
  const hasSel = selIdx >= 0;
  document.getElementById('del-btn').disabled = !hasSel;
  if (hasSel) document.getElementById('class-select').value = boxes[selIdx].cls;
}

function selectedClass() { return Number(document.getElementById('class-select').value); }

function navigate(dir) {
  loadImage(((imgIdx + dir) % images.length + images.length) % images.length);
}

function deleteSelected() {
  if (selIdx < 0) return;
  boxes.splice(selIdx, 1);
  selIdx = Math.min(selIdx, boxes.length - 1);
  if (boxes.length === 0) selIdx = -1;
  updateUI(); redraw(); renderBoxList();
}

async function save() {
  const name = images[imgIdx].name;
  await fetch(`/api/labels/${encodeURIComponent(name)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ boxes }),
  });
  setStatus(`✓ Saved ${boxes.length} boxes`);
  const btn = document.getElementById('save-btn');
  btn.classList.add('saved');
  setTimeout(() => btn.classList.remove('saved'), 1800);

  // mark corrected in sidebar
  images[imgIdx].corrected = true;
  renderImageList();
  document.querySelector(`.img-item[data-idx="${imgIdx}"]`)?.classList.add('active');
}

function setStatus(msg) { document.getElementById('status-msg').textContent = msg; }
init();
</script>
</body>
</html>"""

# inject the corrected folder path into the HTML so user can see it
HTML = HTML.replace("CORRECTED_FOLDER_PLACEHOLDER", str(CORRECTED_FOLDER))

@app.route("/")
def index():
    return HTML

if __name__ == "__main__":
    print("─" * 60)
    print("  🍃 Leaf Annotator")
    print(f"  Images    : {IMAGES_FOLDER}")
    print(f"  Labels in : {LABELS_FOLDER}")
    print(f"  Saving to : {CORRECTED_FOLDER}")
    print("─" * 60)
    print("  Open  http://127.0.0.1:5000")
    print("─" * 60)
    app.run(debug=False, port=5000)