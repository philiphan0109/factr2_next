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
from std_msgs.msg import Float32


HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>FACTR2 NEXT</title>
<style>
body{margin:0;background:#111;color:#ddd;font:14px system-ui}
main{max-width:1100px;margin:28px auto;padding:0 18px}
.top{display:flex;gap:18px;align-items:center;margin-bottom:14px}
h1{font-size:18px;margin:0;color:#fff}.pill{color:#aaa}
canvas{width:100%;height:560px;background:#171717;border:1px solid #333}
.legend{display:flex;gap:18px;flex-wrap:wrap;margin-top:12px}
.item{display:flex;gap:7px;align-items:center}.sw{width:12px;height:12px;border-radius:2px}
label{cursor:pointer} input{accent-color:#f45}
</style></head>
<body><main>
<div class="top"><h1>FACTR2 NEXT</h1><span id="arm" class="pill"></span></div>
<canvas id="plot" width="1100" height="560"></canvas>
<div class="legend" id="legend"></div>
</main>
<script>
const c=document.getElementById("plot"), ctx=c.getContext("2d");
const legend=document.getElementById("legend");
let data={t:[],ext_norm:[],free_norm:[],mse:[],score:[]};
const series=[
  ["ext_norm","|external torque|","#ff5a63",true],
  ["free_norm","|free torque pred|","#58a6ff",true],
  ["mse","mse","#ffb454",true],
  ["score","score","#6ee787",true],
];
function finite(xs){return xs.filter(Number.isFinite)}
function draw(){
  ctx.clearRect(0,0,c.width,c.height);
  const pad={l:60,r:18,t:24,b:44}, W=c.width-pad.l-pad.r, H=c.height-pad.t-pad.b;
  const t=data.t, xs=finite(t); if(xs.length<2) return grid(0,10,0,1);
  const ys=series.flatMap(s=>s[3]?finite(data[s[0]]):[]);
  let xmin=xs[0], xmax=xs[xs.length-1], ymin=Math.min(...ys), ymax=Math.max(...ys);
  if(!Number.isFinite(ymin)||ymin===ymax){ymin=0;ymax=1}
  const margin=(ymax-ymin)*0.08; ymin-=margin; ymax+=margin; grid(xmin,xmax,ymin,ymax);
  for(const [key,label,color,visible] of series){ if(!visible) continue;
    ctx.strokeStyle=color; ctx.lineWidth=2; ctx.beginPath(); let open=false;
    for(let i=0;i<t.length;i++){ const y=data[key][i]; if(!Number.isFinite(y)){open=false;continue}
      const px=pad.l+(t[i]-xmin)/(xmax-xmin)*W, py=pad.t+(1-(y-ymin)/(ymax-ymin))*H;
      open?ctx.lineTo(px,py):ctx.moveTo(px,py); open=true;
    } ctx.stroke();
  }
  legend.innerHTML=series.map(([k,l,color,visible],i)=>{
    const xs=finite(data[k]), v=xs.length?xs[xs.length-1].toFixed(4):"waiting";
    const dim=visible?"":"opacity:.35";
    return `<label class=item style="${dim}"><input type=checkbox data-i="${i}" ${visible?"checked":""}><span class=sw style="background:${color}"></span>${l}: ${v}</label>`;
  }).join("");
  legend.querySelectorAll("input").forEach(x=>x.onchange=()=>{series[x.dataset.i][3]=x.checked; draw();});
  function grid(x0,x1,y0,y1){
    ctx.strokeStyle="#2a2a2a"; ctx.fillStyle="#aaa"; ctx.lineWidth=1; ctx.font="12px system-ui";
    for(let i=0;i<=5;i++){ const x=pad.l+i*W/5, y=pad.t+i*H/5;
      ctx.beginPath(); ctx.moveTo(pad.l,y); ctx.lineTo(pad.l+W,y); ctx.stroke();
      ctx.fillText((y1-(y1-y0)*i/5).toFixed(2),8,y+4);
      ctx.beginPath(); ctx.moveTo(x,pad.t); ctx.lineTo(x,pad.t+H); ctx.stroke();
      ctx.fillText((x0+(x1-x0)*i/5).toFixed(1),x-10,pad.t+H+24);
    }
  }
}
new EventSource("/events").onmessage=(ev)=>{data=JSON.parse(ev.data); document.getElementById("arm").textContent=data.arm; draw();};
</script></body></html>
"""


class WebNode(Node):
    def __init__(self):
        super().__init__("factr2_next_visualize")
        default = Path(get_package_share_directory("factr2_next")) / "config" / "visualize.yaml"
        self.cfg = self._load_config(self.declare_parameter("config_file", str(default)).value)
        self.arm = str(self.cfg.get("arm", "right"))
        self.max_points = int(self.cfg.get("plot", {}).get("max_points", 500))
        self.data = {k: deque(maxlen=self.max_points) for k in ("t", "ext_norm", "free_norm", "mse", "score")}
        self.latest = {"ext_norm": np.nan, "free_norm": np.nan, "mse": np.nan, "score": np.nan}
        self.t0, self.seq, self.running = time.monotonic(), 0, True
        self.cond = threading.Condition()

        self._subscribe()
        port = int(self.cfg.get("web", {}).get("port", 8080))
        self.httpd = ThreadingHTTPServer(("0.0.0.0", port), self._handler())
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.get_logger().info(f"NEXT plot at http://localhost:{port}")

    def _subscribe(self):
        outputs = self.cfg["outputs"]
        self.subs = [
            self.create_subscription(JointState, self._topic(outputs["external_joint_torque"]),
                                     lambda m: self._set_norm("ext_norm", m.position, True), qos_profile_sensor_data),
            self.create_subscription(JointState, self._topic(outputs["free_joint_torque_pred"]),
                                     lambda m: self._set_norm("free_norm", m.position), qos_profile_sensor_data),
            self.create_subscription(Float32, self._topic(outputs["mse"]),
                                     lambda m: self._set_scalar("mse", m.data), 10),
            self.create_subscription(Float32, self._topic(outputs["score"]),
                                     lambda m: self._set_scalar("score", m.data), 10),
        ]

    def _set_norm(self, key, values, sample=False):
        self.latest[key] = float(np.linalg.norm(np.asarray(values, dtype=float)))
        if sample:
            self._append()

    def _set_scalar(self, key, value):
        self.latest[key] = float(value)

    def _append(self):
        with self.cond:
            self.data["t"].append(time.monotonic() - self.t0)
            for key in ("ext_norm", "free_norm", "mse", "score"):
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
