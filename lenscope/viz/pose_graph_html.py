"""Self-contained HTML visualization of a sampled scene (for visual inspection).

Zero dependencies (pure canvas 2D): top-down rooms/portals/furniture, poses colored
by kind (graph/bridge/supervision) with yaw arrows, covisibility edges alpha-weighted,
hover for pose info, edge-threshold slider re-filters live. Open in any browser.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..core.spec import SceneSpec

_HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>body{background:#101216;color:#e8ebf0;font:13px system-ui;margin:0}
#c{display:block}#hud{position:fixed;top:10px;left:10px;background:rgba(18,21,28,.9);
border:1px solid #2a2f3a;border-radius:8px;padding:10px 12px;max-width:330px}
input[type=range]{width:160px;vertical-align:middle}</style></head><body>
<canvas id="c"></canvas><div id="hud"><b>__TITLE__</b><br>
edge&ge;<span id="tv">0.30</span> <input id="th" type="range" min="0" max="0.9" step="0.05" value="0.30"><br>
<span id="stats"></span><br><span id="info" style="color:#9aa3b0">hover a pose…</span></div>
<script>
const D = __DATA__;
const cv = document.getElementById('c'), ctx = cv.getContext('2d');
let TH = 0.30;
function fit(){cv.width=innerWidth;cv.height=innerHeight;draw();}
addEventListener('resize', fit);
let xs=[].concat(...D.rooms.map(r=>[r.x0,r.x1])), ys=[].concat(...D.rooms.map(r=>[r.y0,r.y1]));
const mnx=Math.min(...xs)-1, mxx=Math.max(...xs)+1, mny=Math.min(...ys)-1, mxy=Math.max(...ys)+1;
function S(){return Math.min(cv.width/(mxx-mnx), cv.height/(mxy-mny));}
function X(x){return (x-mnx)*S()+(cv.width-(mxx-mnx)*S())/2;}
function Y(y){return cv.height-((y-mny)*S()+(cv.height-(mxy-mny)*S())/2);}
function draw(){
 ctx.clearRect(0,0,cv.width,cv.height);
 for(const r of D.rooms){ctx.strokeStyle='#5a6273';ctx.lineWidth=2;
  ctx.strokeRect(X(r.x0),Y(r.y1),(r.x1-r.x0)*S(),(r.y1-r.y0)*S());}
 for(const f of D.furniture){ctx.fillStyle='rgba(120,130,150,.25)';
  ctx.fillRect(X(f.cx-f.sx/2),Y(f.cy+f.sy/2),f.sx*S(),f.sy*S());}
 for(const p of D.portals){ctx.strokeStyle=p.kind==='window'?'#4da3ff':'#ffcc44';ctx.lineWidth=4;
  ctx.beginPath();ctx.moveTo(X(p.x0),Y(p.y0));ctx.lineTo(X(p.x1),Y(p.y1));ctx.stroke();}
 let ne=0;
 for(const e of D.edges) if(e[2]>=TH){ne++;
  ctx.strokeStyle='rgba(85,220,140,'+Math.min(e[2],0.9)+')';ctx.lineWidth=1;
  const A=D.poses[e[0]],B=D.poses[e[1]];
  ctx.beginPath();ctx.moveTo(X(A.pos[0]),Y(A.pos[1]));ctx.lineTo(X(B.pos[0]),Y(B.pos[1]));ctx.stroke();}
 for(const p of D.poses){
  ctx.fillStyle=p.kind==='bridge'?'#ffcc44':(p.supervision_only?'#cc66ff':'#ff5555');
  ctx.beginPath();ctx.arc(X(p.pos[0]),Y(p.pos[1]),4,0,7);ctx.fill();
  ctx.strokeStyle='#e8ebf0';ctx.lineWidth=1;ctx.beginPath();
  ctx.moveTo(X(p.pos[0]),Y(p.pos[1]));
  ctx.lineTo(X(p.pos[0]+0.45*Math.cos(p.yaw)),Y(p.pos[1]+0.45*Math.sin(p.yaw)));ctx.stroke();}
 document.getElementById('stats').textContent =
  D.poses.length+' poses ('+D.report.bridge_poses+' bridge) · '+ne+' edges · min_deg '
  +D.report.min_degree+' · λ2 '+D.report.lambda2+' · sup '+D.report.supervision_share;
}
document.getElementById('th').oninput=function(e){TH=+e.target.value;
 document.getElementById('tv').textContent=TH.toFixed(2);draw();};
cv.onmousemove=function(e){
 for(const p of D.poses){const dx=e.clientX-X(p.pos[0]),dy=e.clientY-Y(p.pos[1]);
  if(dx*dx+dy*dy<64){document.getElementById('info').textContent=
   '#'+p.id+' '+p.kind+' room'+p.room+' h='+p.pos[2]+' hcls='+p.height_class+
   (p.supervision_only?' [SUP]':'');return;}}
};
fit();
</script></body></html>"""


def write_html(spec: SceneSpec, result: dict, out: Path):
    data = {"rooms": [vars(r) for r in spec.rooms],
            "portals": [vars(p) for p in spec.portals],
            "furniture": [vars(f) for f in spec.furniture],
            "poses": result["poses"], "edges": result["edges"], "report": result["report"]}
    html = _HTML.replace("__TITLE__", spec.name).replace("__DATA__", json.dumps(data))
    Path(out).write_text(html)
    return out
