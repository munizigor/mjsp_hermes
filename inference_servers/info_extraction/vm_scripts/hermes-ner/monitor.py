from flask import Flask, jsonify
import subprocess
import psutil

app = Flask(__name__)

@app.route("/metrics")
def gpu():
    out = subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits"
    ]).decode().strip()

    util, used, total = map(int, out.split(", "))
    # interval=0.1 calculates the usage over a tenth of a second for an accurate reading. 
    # Without an interval, the first call will return 0.0.
    cpu_percent = psutil.cpu_percent(interval=0.1)
    
    # virtual_memory() returns a named tuple containing system memory usage statistics in bytes.
    mem_info = psutil.virtual_memory()
   
    return jsonify({
        "gpu_utilization_percent": util,
        "vram_used_mb": used,
        "vram_total_mb": total,
	    "cpu_usage_percent": cpu_percent,
        "ram_used_bytes": mem_info.used,
        "ram_total_bytes": mem_info.total,
    })

app.run(host="0.0.0.0", port=8002)
