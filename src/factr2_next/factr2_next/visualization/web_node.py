import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32


HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>FACTR2 NEXT</title>
<style>
:root{color-scheme:dark;--bg:#101214;--panel:#181b1f;--line:#2b3036;--text:#e8edf2;--muted:#98a2ad}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--text);font:14px system-ui,sans-serif}
main{max-width:1280px;margin:22px auto;padding:0 18px}.top{display:flex;align-items:end;justify-content:space-between;gap:18px;margin-bottom:14px}
h1{font-size:20px;margin:0}.pill{color:var(--muted);font-size:13px}.plot{background:#14171a;border:1px solid var(--line);border-radius:8px;overflow:hidden}
canvas{display:block;width:100%;height:580px}.controls{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;margin-top:12px}
.group{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px}.group h2{font-size:12px;font-weight:650;color:var(--muted);margin:0 0 8px;text-transform:uppercase}
.items{display:flex;flex-wrap:wrap;gap:7px}.item{display:flex;align-items:center;gap:6px;border:1px solid var(--line);border-radius:6px;padding:5px 7px;cursor:pointer;user-select:none}
.item.off{opacity:.42}.sw{width:10px;height:10px;border-radius:2px;flex:0 0 auto}input{accent-color:#4ea1ff;margin:0}.value{color:var(--muted);font-variant-numeric:tabular-nums}.status{margin-top:8px;color:var(--muted);font-size:12px}
</style></head>
<body><main>
<div class="top"><div><h1>FACTR2 NEXT</h1><div class="pill" id="arm">waiting</div></div><div class="pill" id="status">connecting</div></div>
<div class="plot"><canvas id="plot" width="1240" height="580"></canvas></div>
<div class="controls" id="controls"></div>
</main>
<script>
const canvas=document.getElementById("plot"), ctx=canvas.getContext("2d");
const controls=document.getElementById("controls"), statusEl=document.getElementById("status");
let data={t:[]}, controlsReady=false;
const groups=[
  ["Summary",[
    ["ext_norm","|tau ext| filtered","#ff5a63",true],
    ["ext_raw_norm","|tau ext| raw","#f7a1a8",false],
    ["fb_norm","|tau fb out|","#9cdcfe",true],
    ["fb_gate","feedback gate","#c586c0",true],
    ["free_norm","|tau free pred|","#58a6ff",false],
    ["mse","mse","#ffb454",false],
    ["score","score","#6ee787",true],
    ["contact","contact","#d2a8ff",true],
  ]],
  ["Filtered joints",[
    ["ext_j1","j1","#ff6b6b",false],["ext_j2","j2","#ffa94d",false],["ext_j3","j3","#ffd43b",false],
    ["ext_j4","j4","#69db7c",false],["ext_j5","j5","#4dabf7",false],["ext_j6","j6","#b197fc",false],
  ]],
  ["Raw joints",[
    ["raw_j1","j1 raw","#ff8787",false],["raw_j2","j2 raw","#ffc078",false],["raw_j3","j3 raw","#ffe066",false],
    ["raw_j4","j4 raw","#8ce99a",false],["raw_j5","j5 raw","#74c0fc",false],["raw_j6","j6 raw","#c4b5fd",false],
  ]],
  ["Feedback output",[
    ["fb_j1","j1 fb","#7dd3fc",false],["fb_j2","j2 fb","#67e8f9",false],["fb_j3","j3 fb","#5eead4",false],
    ["fb_j4","j4 fb","#86efac",false],["fb_j5","j5 fb","#fde047",false],["fb_j6","j6 fb","#f0abfc",false],
  ]],
];
const series=groups.flatMap(g=>g[1]).map(s=>({key:s[0],label:s[1],color:s[2],visible:loadVisible(s[0],s[3])}));
function loadVisible(key, fallback){const v=localStorage.getItem("next."+key); return v===null?fallback:v==="1"}
function setVisible(key, value){localStorage.setItem("next."+key,value?"1":"0")}
function finite(xs){return (xs||[]).filter(Number.isFinite)}
function setupControls(){
  controls.innerHTML=groups.map(([name,items])=>`<section class=group><h2>${name}</h2><div class=items>${
    items.map(([key,label,color])=>{
      const s=series.find(x=>x.key===key), checked=s.visible?"checked":"";
      return `<label class="item ${s.visible?"":"off"}" data-key="${key}"><input type=checkbox ${checked}><span class=sw style="background:${color}"></span><span>${label}</span><span class=value id="v-${key}">--</span></label>`;
    }).join("")
  }</div></section>`).join("");
  controls.querySelectorAll(".item").forEach(item=>{
    const key=item.dataset.key, box=item.querySelector("input"), s=series.find(x=>x.key===key);
    box.onchange=()=>{s.visible=box.checked; item.classList.toggle("off",!s.visible); setVisible(key,s.visible); draw();};
  });
  controlsReady=true;
}
function draw(){
  if(!controlsReady) setupControls();
  ctx.clearRect(0,0,canvas.width,canvas.height);
  const pad={l:62,r:18,t:22,b:42}, W=canvas.width-pad.l-pad.r, H=canvas.height-pad.t-pad.b;
  const t=data.t||[], xs=finite(t);
  const ys=series.flatMap(s=>s.visible?finite(data[s.key]):[]);
  let xmin=xs.length?xs[0]:0, xmax=xs.length?xs[xs.length-1]:10, ymin=Math.min(...ys), ymax=Math.max(...ys);
  if(xs.length<2){xmax=10}
  if(!Number.isFinite(ymin)||!Number.isFinite(ymax)||ymin===ymax){ymin=-1;ymax=1}
  const m=(ymax-ymin)*0.08; ymin-=m; ymax+=m; grid(pad,W,H,xmin,xmax,ymin,ymax);
  for(const s of series){ if(!s.visible) continue; drawLine(s,pad,W,H,xmin,xmax,ymin,ymax); }
  for(const s of series){ const el=document.getElementById("v-"+s.key), vals=finite(data[s.key]); if(el) el.textContent=vals.length?vals[vals.length-1].toFixed(3):"--"; }
}
function drawLine(s,pad,W,H,xmin,xmax,ymin,ymax){
  const t=data.t||[], vals=data[s.key]||[]; ctx.strokeStyle=s.color; ctx.lineWidth=2; ctx.beginPath(); let open=false;
  for(let i=0;i<t.length;i++){const y=vals[i]; if(!Number.isFinite(y)){open=false;continue}
    const px=pad.l+(t[i]-xmin)/Math.max(1e-6,xmax-xmin)*W, py=pad.t+(1-(y-ymin)/(ymax-ymin))*H;
    open?ctx.lineTo(px,py):ctx.moveTo(px,py); open=true;
  } ctx.stroke();
}
function grid(pad,W,H,x0,x1,y0,y1){
  ctx.strokeStyle="#293039"; ctx.fillStyle="#9aa4ae"; ctx.lineWidth=1; ctx.font="12px system-ui";
  for(let i=0;i<=5;i++){const x=pad.l+i*W/5, y=pad.t+i*H/5;
    ctx.beginPath(); ctx.moveTo(pad.l,y); ctx.lineTo(pad.l+W,y); ctx.stroke();
    ctx.fillText((y1-(y1-y0)*i/5).toFixed(2),8,y+4);
    ctx.beginPath(); ctx.moveTo(x,pad.t); ctx.lineTo(x,pad.t+H); ctx.stroke();
    ctx.fillText((x0+(x1-x0)*i/5).toFixed(1),x-8,pad.t+H+24);
  }
}
new EventSource("/events").onmessage=(ev)=>{
  data=JSON.parse(ev.data); document.getElementById("arm").textContent=`arm: ${data.arm}  samples: ${(data.t||[]).length}`;
  statusEl.textContent="live"; draw();
};
setupControls(); draw();
</script></body></html>
"""


class WebNode(Node):
    def __init__(self):
        super().__init__("factr2_next_visualize")
        default = Path(get_package_share_directory("factr2_next")) / "config" / "visualize.yaml"
        self.cfg = self._load_config(self.declare_parameter("config_file", str(default)).value)
        self.arm = str(self.cfg.get("arm", "right"))
        self.max_points = int(self.cfg.get("plot", {}).get("max_points", 500))
        self.keys = (
            "t",
            "ext_norm",
            "ext_raw_norm",
            "fb_norm",
            "fb_gate",
            "free_norm",
            "mse",
            "score",
            "contact",
            "ext_j1",
            "ext_j2",
            "ext_j3",
            "ext_j4",
            "ext_j5",
            "ext_j6",
            "raw_j1",
            "raw_j2",
            "raw_j3",
            "raw_j4",
            "raw_j5",
            "raw_j6",
            "fb_j1",
            "fb_j2",
            "fb_j3",
            "fb_j4",
            "fb_j5",
            "fb_j6",
        )
        self.data = {key: deque(maxlen=self.max_points) for key in self.keys}
        self.latest = {key: np.nan for key in self.keys if key != "t"}
        self.t0, self.seq, self.running = time.monotonic(), 0, True
        self.cond = threading.Condition()

        self._subscribe()
        port = int(self.cfg.get("web", {}).get("port", 8080))
        self.httpd = ThreadingHTTPServer(("0.0.0.0", port), self._handler())
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.get_logger().info(f"NEXT plot at http://localhost:{port}")

    def _subscribe(self):
        outputs = self.cfg["outputs"]
        raw_topic = outputs.get(
            "external_joint_torque_raw",
            outputs["external_joint_torque"] + "/raw",
        )
        self.subs = [
            self.create_subscription(
                JointState,
                self._topic(outputs["external_joint_torque"]),
                lambda m: self._set_torque("ext", m.position, True),
                qos_profile_sensor_data,
            ),
            self.create_subscription(
                JointState,
                self._topic(raw_topic),
                lambda m: self._set_torque("raw", m.position),
                qos_profile_sensor_data,
            ),
            self.create_subscription(
                JointState,
                self._topic(outputs["free_joint_torque_pred"]),
                lambda m: self._set_norm("free_norm", m.position),
                qos_profile_sensor_data,
            ),
            self.create_subscription(
                Float32,
                self._topic(outputs["mse"]),
                lambda m: self._set_scalar("mse", m.data),
                10,
            ),
            self.create_subscription(
                Float32,
                self._topic(outputs["score"]),
                lambda m: self._set_scalar("score", m.data),
                10,
            ),
        ]
        if "contact_state" in outputs:
            self.subs.append(
                self.create_subscription(
                    Bool,
                    self._topic(outputs["contact_state"]),
                    lambda m: self._set_scalar("contact", 1.0 if m.data else 0.0),
                    10,
                )
            )
        if "feedback_torque" in outputs:
            self.subs.append(
                self.create_subscription(
                    JointState,
                    self._topic(outputs["feedback_torque"]),
                    lambda m: self._set_torque("fb", m.position),
                    qos_profile_sensor_data,
                )
            )
        if "feedback_gate" in outputs:
            self.subs.append(
                self.create_subscription(
                    Float32,
                    self._topic(outputs["feedback_gate"]),
                    lambda m: self._set_scalar("fb_gate", m.data),
                    10,
                )
            )

    def _set_norm(self, key, values, sample=False):
        self.latest[key] = float(np.linalg.norm(np.asarray(values, dtype=float)))
        if sample:
            self._append()

    def _set_torque(self, prefix, values, sample=False):
        values = np.asarray(values, dtype=float)
        keys = {
            "ext": ("ext_norm", "ext"),
            "raw": ("ext_raw_norm", "raw"),
            "fb": ("fb_norm", "fb"),
        }
        norm_key, joint_prefix = keys[prefix]
        self.latest[norm_key] = float(np.linalg.norm(values))
        for i in range(6):
            self.latest[f"{joint_prefix}_j{i + 1}"] = float(values[i]) if i < len(values) else np.nan
        if sample:
            self._append()

    def _set_scalar(self, key, value):
        self.latest[key] = float(value)

    def _append(self):
        with self.cond:
            self.data["t"].append(time.monotonic() - self.t0)
            for key in self.keys:
                if key == "t":
                    continue
                self.data[key].append(self.latest[key])
            self.seq += 1
            self.cond.notify_all()

    def _snapshot(self):
        with self.cond:
            out = {k: [x if np.isfinite(x) else None for x in v] for k, v in self.data.items()}
            out["arm"] = self.arm
            return json.dumps(out).encode()

    def _handler(self):
        node = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                if self.path == "/events":
                    self._events()
                    return
                body = HTML.encode()
                self.send_response(200)
                self.send_header("content-type", "text/html")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _events(self):
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.send_header("cache-control", "no-cache")
                self.end_headers()
                last = -1
                while node.running and rclpy.ok():
                    with node.cond:
                        node.cond.wait_for(lambda: node.seq != last or not node.running, 1.0)
                        last = node.seq
                    try:
                        self.wfile.write(b"data: " + node._snapshot() + b"\n\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        break

        return Handler

    def destroy_node(self):
        self.running = False
        with self.cond:
            self.cond.notify_all()
        self.httpd.shutdown()
        super().destroy_node()

    def _topic(self, topic):
        return str(topic).format(arm=self.arm)

    def _load_config(self, path):
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}


def main(args=None):
    rclpy.init(args=args)
    node = WebNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
