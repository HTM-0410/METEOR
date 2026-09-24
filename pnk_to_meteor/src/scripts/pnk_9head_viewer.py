#!/usr/bin/env python3
"""Interactive local viewer for processed PNK 9-head scenes and frames."""
from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.visualize_pnk_layer2_gt_samples import render_sample  # noqa: E402


DEFAULT_ROOT = Path(r"D:\Backup_rosbag\PNKData_layer2_9head_v7")
DEFAULT_CACHE = REPO / "out" / "pnk_9head_v7_viewer_cache"


HTML = r"""<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PNK Layer 2 · 9-head Viewer</title>
<style>
:root{color-scheme:dark;--bg:#0a0d12;--card:#111722;--line:#263143;--text:#edf3fb;--muted:#9cacbf;--accent:#4cc2ff;--ok:#4bd28b;--warn:#ffca5c}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.4 Inter,Segoe UI,Arial,sans-serif}
header{position:sticky;top:0;z-index:4;background:#0c1119ee;backdrop-filter:blur(12px);border-bottom:1px solid var(--line);padding:14px 20px}
h1{font-size:18px;margin:0 0 3px}.sub{color:var(--muted);font-size:12px}.layout{display:grid;grid-template-columns:300px minmax(0,1fr);min-height:calc(100vh - 68px)}
aside{border-right:1px solid var(--line);padding:16px;background:#0c1118}.main{padding:18px;min-width:0}
label{display:block;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.08em;margin:14px 0 6px}
select,input[type=number]{width:100%;background:#151d29;color:var(--text);border:1px solid #344258;border-radius:8px;padding:9px}
input[type=range]{width:100%;accent-color:var(--accent)}.row{display:flex;gap:8px;align-items:center}.row>*{flex:1}
button{background:#182536;border:1px solid #344258;color:var(--text);border-radius:8px;padding:9px 12px;cursor:pointer}button:hover{border-color:var(--accent)}
.badges{display:flex;flex-wrap:wrap;gap:6px;margin-top:12px}.badge{padding:4px 7px;border-radius:999px;background:#1b2737;color:#bfd0e4;font-size:11px}.badge.ok{color:#b5f7d2;background:#123425}.badge.off{color:#ffe3a8;background:#392c12}
.stats{margin-top:14px;border-top:1px solid var(--line);padding-top:10px}.stat{display:flex;justify-content:space-between;gap:12px;padding:5px 0;border-bottom:1px solid #182231}.stat span:first-child{color:var(--muted)}
.toolbar{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:12px}.status{color:var(--muted)}
.viewer{background:#05070a;border:1px solid var(--line);border-radius:12px;overflow:auto;min-height:460px;display:flex;align-items:flex-start;justify-content:center}
.viewer img{display:block;max-width:none;width:min(100%,1260px);height:auto;cursor:zoom-in}.viewer img.zoom{width:1260px}
.help{margin-top:10px;color:var(--muted);font-size:12px}.error{color:#ff8d8d;white-space:pre-wrap;padding:30px}
@media(max-width:900px){.layout{grid-template-columns:1fr}aside{border-right:0;border-bottom:1px solid var(--line)}header{position:relative}}
</style></head>
<body>
<header><h1>PNK Layer 2 · 9-head Viewer</h1><div class="sub" id="datasetLine">Đang tải dataset…</div></header>
<div class="layout">
<aside>
  <label>Scene</label><select id="scene"></select>
  <label>Frame</label><input id="slider" type="range" min="0" value="0"><div class="row"><button id="prev">← Trước</button><input id="frame" type="number" min="0" value="0"><button id="next">Sau →</button></div>
  <div class="badges" id="badges"></div>
  <div class="stats" id="stats"></div>
</aside>
<main class="main">
  <div class="toolbar"><strong id="title">Sample</strong><span class="status" id="status">Sẵn sàng</span></div>
  <div class="viewer" id="viewer"><img id="image" alt="PNK Layer 2 visualization"></div>
  <div class="help">Phím ←/→ đổi frame · nhấp ảnh để zoom 1:1 · ảnh được render và cache khi mở lần đầu.</div>
</main></div>
<script>
const state={doc:null,scene:null,request:0};
const $=id=>document.getElementById(id);
function esc(s){return encodeURIComponent(s)}
async function boot(){
  const r=await fetch('/api/scenes'); state.doc=await r.json();
  $('datasetLine').textContent=`${state.doc.dataset} · ${state.doc.frames.toLocaleString()} frames · heads ${state.doc.active_heads.join(', ')}`;
  $('scene').innerHTML=state.doc.scenes.map((s,i)=>`<option value="${i}">${s.scene} · ${s.frames.length} frames</option>`).join('');
  setScene(0);
}
function setScene(index){state.scene=state.doc.scenes[index];const n=state.scene.frames.length;$('slider').max=n-1;$('frame').max=n-1;$('slider').value=0;$('frame').value=0;loadIndex(0)}
function clamp(v){return Math.max(0,Math.min(state.scene.frames.length-1,Number(v)||0))}
async function loadIndex(v){
  const i=clamp(v);$('slider').value=i;$('frame').value=i;const f=state.scene.frames[i];const rid=++state.request;
  $('status').textContent='Đang đọc metadata…';$('title').textContent=`${state.scene.scene} · frame ${f.frame}`;
  try{
    const meta=await (await fetch(`/api/frame?scene=${esc(state.scene.scene)}&frame=${f.frame}`)).json();if(rid!==state.request)return;
    const valid=meta.supervision_valid||{};$('badges').innerHTML=Object.entries(valid).map(([k,v])=>`<span class="badge ${v?'ok':'off'}">${k}: ${v?'valid':'off'}</span>`).join('');
    const lane=meta.bev_lane_class_pixels||{};
    const rows={route_command:meta.driving_command_label,timestamp_ns:meta.timestamp_ns,source_boxes:meta.box3d_source_count,profile_boxes:meta.box3d_profile_count,tier_a:meta.box3d_tier_a_count,tier_b:meta.box3d_tier_b_count,depth_measured:meta.depth_measured_pixels,depth_guided:meta.depth_interpolated_pixels,depth_total:meta.depth_valid_pixels,occupancy_original:meta.occupancy_original_observed_voxels,occupancy_added:meta.occupancy_temporal_added_voxels,drivable_original:meta.drivable_original_pixels,drivable_added:meta.drivable_temporal_added_pixels,crosswalk:lane.crosswalk,laneline:lane.laneline,stopline:lane.stopline,road_edge:lane.road_edge,marking:lane.marking,parking:lane.parking};
    $('stats').innerHTML=Object.entries(rows).filter(x=>x[1]!==undefined).map(([k,v])=>`<div class="stat"><span>${k}</span><strong>${typeof v==='number'?Number(v).toLocaleString():v}</strong></div>`).join('');
    $('status').textContent='Đang render/cache ảnh…';
    const img=$('image');img.style.opacity='.35';img.onload=()=>{if(rid===state.request){$('status').textContent='Đã render';img.style.opacity='1'}};img.onerror=()=>{if(rid===state.request){$('status').textContent='Lỗi render';$('viewer').innerHTML='<div class="error">Không render được frame.</div>'}};
    img.src=`/render?scene=${esc(state.scene.scene)}&frame=${f.frame}&v=${Date.now()}`;
  }catch(e){$('status').textContent='Lỗi';$('viewer').innerHTML=`<div class="error">${e}</div>`}
}
$('scene').onchange=e=>setScene(Number(e.target.value));$('slider').oninput=e=>loadIndex(e.target.value);$('frame').onchange=e=>loadIndex(e.target.value);
$('prev').onclick=()=>loadIndex(Number($('frame').value)-1);$('next').onclick=()=>loadIndex(Number($('frame').value)+1);
$('image').onclick=e=>e.target.classList.toggle('zoom');window.onkeydown=e=>{if(e.key==='ArrowLeft')$('prev').click();if(e.key==='ArrowRight')$('next').click()};
boot().catch(e=>{$('datasetLine').textContent='Không tải được dataset: '+e});
</script></body></html>"""


class ViewerState:
    def __init__(self, root: Path, cache: Path):
        self.root = root.resolve(); self.cache = cache.resolve()
        self.dataset = json.loads((self.root / "dataset.json").read_text(encoding="utf-8"))
        self.manifests = {}
        self.frames = {}
        scenes = []
        for path in sorted(self.root.glob("*/manifest.json")):
            manifest = json.loads(path.read_text(encoding="utf-8"))
            scene = manifest["scene"]
            self.manifests[scene] = manifest
            self.frames[scene] = {int(f["frame"]): f for f in manifest["frames"]}
            scenes.append({"scene": scene,
                           "frames": [{"frame": int(f["frame"]),
                                       "timestamp_ns": int(f["timestamp_ns"])}
                                      for f in manifest["frames"]]})
        self.scene_doc = {
            "dataset": self.root.name,
            "schema": self.dataset.get("schema"),
            "frames": sum(len(x["frames"]) for x in scenes),
            "active_heads": self.dataset.get("active_heads", []),
            "excluded_heads": self.dataset.get("excluded_heads", {}),
            "scenes": scenes,
        }
        self.lock = threading.Lock()

    def frame(self, scene: str, frame: int):
        try:
            return self.frames[scene][frame]
        except KeyError as exc:
            raise ValueError("Unknown scene/frame") from exc

    def render(self, scene: str, frame: int) -> Path:
        self.frame(scene, frame)
        destination = self.cache / scene / f"{frame:06d}.jpg"
        if destination.is_file():
            return destination
        with self.lock:
            if not destination.is_file():
                render_sample(self.root,
                              {"scene": scene, "frame": frame,
                               "selection_reason": "interactive_viewer"},
                              destination)
        return destination


def handler_factory(state: ViewerState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            sys.stdout.write("[viewer] " + fmt % args + "\n"); sys.stdout.flush()

        def send_bytes(self, data: bytes, content_type: str, status=HTTPStatus.OK):
            self.send_response(status); self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data))); self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                # A fast frame change cancels the previous image request.  The
                # render is still cached and the next request remains valid.
                pass

        def send_json(self, value, status=HTTPStatus.OK):
            self.send_bytes(json.dumps(value, ensure_ascii=False).encode("utf-8"),
                            "application/json; charset=utf-8", status)

        def do_GET(self):
            parsed = urlparse(self.path); query = parse_qs(parsed.query)
            try:
                if parsed.path == "/":
                    return self.send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
                if parsed.path == "/health":
                    return self.send_json({"status": "ok", "dataset": str(state.root)})
                if parsed.path == "/api/scenes":
                    return self.send_json(state.scene_doc)
                if parsed.path == "/api/frame":
                    scene = query.get("scene", [""])[0]; frame = int(query.get("frame", ["-1"])[0])
                    return self.send_json(state.frame(scene, frame))
                if parsed.path == "/render":
                    scene = query.get("scene", [""])[0]; frame = int(query.get("frame", ["-1"])[0])
                    path = state.render(scene, frame)
                    return self.send_bytes(path.read_bytes(), mimetypes.guess_type(path)[0] or "image/jpeg")
                return self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except (ValueError, KeyError) as exc:
                return self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except Exception as exc:
                return self.send_json({"error": repr(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    state = ViewerState(args.root, args.cache)
    server = ThreadingHTTPServer((args.host, args.port), handler_factory(state))
    print(f"PNK viewer: http://{args.host}:{args.port}", flush=True)
    print(f"Dataset: {state.root} ({state.scene_doc['frames']} frames)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
