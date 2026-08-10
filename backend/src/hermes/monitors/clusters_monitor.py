import os
from shutil import copyfile
from glob import glob
import time
import multiprocessing as mp
import sys
import json
import requests
from copy import deepcopy
import sqlite3 as sqlite

PROMETHEUS_READINGS_DIR = "/prometheus_readings"

prometheus_columns = [
    "cpu_utilization",
    "cpu_memory_total_bytes",
    "cpu_memory_used_bytes",
    "gpu_memory_total_bytes",
    "gpu_memory_used_bytes",
    "gpu_utilization",
]


class TritonMonitor:
    def __init__(
        self,
        hostname,
        server_type,
        port=8002,
        poll_interval_ms=100,
        file_stable_seconds=0.4,
    ):
        self.hostname = hostname
        self.type = server_type
        self.port = port
        self.poll_interval_ms = poll_interval_ms
        self.file_stable_seconds = file_stable_seconds
        self.stop_signal = mp.Value("H", 0)
        print("Monitor created!", file=sys.stderr)

    def get_system_metrics(self) -> dict:
        url = f"http://{self.hostname}:{self.port}/metrics"
        try:
            response = requests.get(url, timeout=5000)
            response.raise_for_status()
        except Exception as err:
            print(
                f"TritonMonitor: Failed to get metrics from {url}",
                "Error:",
                err,
                file=sys.stderr,
            )
            return {}

        # Dicionário inicial zerado
        data = {
            "cpu_utilization": 0.0,
            "cpu_memory_total_bytes": 0.0,
            "cpu_memory_used_bytes": 0.0,
            "gpu_memory_total_bytes": 0.0,
            "gpu_memory_used_bytes": 0.0,
            "gpu_utilization": 0.0,
        }

        try:
            json_resp = json.loads(response.text)
            if "cpu_usage_percent" in json_resp:
                cpu_raw = json_resp["cpu_usage_percent"]
                data["cpu_utilization"] = float(cpu_raw) / 100
            if "gpu_utilization_percent" in json_resp:
                gpu_raw = json_resp["gpu_utilization_percent"]
                data["gpu_utilization"] = float(gpu_raw) / 100
            if "ram_total_bytes" in json_resp:
                data["cpu_memory_total_bytes"] = float(json_resp["ram_total_bytes"])
            if "ram_used_bytes" in json_resp:
                data["cpu_memory_used_bytes"] = float(json_resp["ram_used_bytes"])
            if "vram_total_mb" in json_resp:
                val_mb = float(json_resp["vram_total_mb"])
                data["gpu_memory_total_bytes"] = val_mb * 1024 * 1024
            if "vram_used_mb" in json_resp:
                val_mb = float(json_resp["vram_used_mb"])
                data["gpu_memory_used_bytes"] = val_mb * 1024 * 1024
        except json.JSONDecodeError as err:
            """print(
                f"Failed to decode JSON from {url}, using custom metrics server: {err}.",
                file=sys.stderr,
            )
            print("Defaulting to Triton metrics reading", file=sys.stderr)"""

            # Lista temporária para calcular a média de utilização das GPUs
            gpu_util_values = []

            for line in response.text.splitlines():
                if line.startswith("#") or not line.strip():
                    continue

                # Separa a chave (com labels) do valor
                key_part, value_part = line.rsplit(" ", 1)

                try:
                    value = float(value_part)
                except ValueError:
                    continue

                # Lógica de extração e agregação
                if "cpu_utilization" in key_part:
                    data["cpu_utilization"] = value

                elif "cpu_memory_total_bytes" in key_part:
                    data["cpu_memory_total_bytes"] = value

                elif "cpu_memory_used_bytes" in key_part:
                    data["cpu_memory_used_bytes"] = value

                elif "gpu_memory_total_bytes" in key_part:
                    data["gpu_memory_total_bytes"] += value  # Soma

                elif "gpu_memory_used_bytes" in key_part:
                    data["gpu_memory_used_bytes"] += value  # Soma

                elif "gpu_utilization" in key_part:
                    gpu_util_values.append(value)  # Guarda para média

            # Calcula a média da utilização de GPU se houver GPUs detectadas
            if gpu_util_values:
                data["gpu_utilization"] = sum(gpu_util_values) / len(gpu_util_values)

        return data


class VLLMMonitor:
    def __init__(
        self,
        hostname,
        server_type,
        port=8002,
        poll_interval_ms=100,
        file_stable_seconds=0.4,
    ):
        self.hostname = hostname
        self.type = server_type
        self.port = port
        self.poll_interval_ms = poll_interval_ms
        self.file_stable_seconds = file_stable_seconds
        self.stop_signal = mp.Value("H", 0)
        print(
            "VLLMMonitor created for!",
            self.hostname,
            self.type,
            self.port,
            file=sys.stderr,
        )

    def get_system_metrics(self) -> dict:
        """
        (base) pita@hermes-backend:~/repos/hermes-agents$ curl 10.221.112.10:8002/metrics
            {"cpu_usage_percent":0.0,"gpu_utilization_percent":0,
            "ram_total_bytes":31535259648,"ram_used_bytes":4435914752,
            "vram_total_mb":15360,"vram_used_mb":14833}
        """
        url = f"http://{self.hostname}:{self.port}/metrics"
        try:
            response = requests.get(url, timeout=5000)
            response.raise_for_status()
        except requests.RequestException as err:
            print(
                "VLLMMonitor: Failed to get metrics from",
                url,
                "Error:",
                err,
                file=sys.stderr,
            )
            return {}

        # Dicionário inicial zerado
        data = {
            "cpu_utilization": 0.0,
            "cpu_memory_total_bytes": 0.0,
            "cpu_memory_used_bytes": 0.0,
            "gpu_memory_total_bytes": 0.0,
            "gpu_memory_used_bytes": 0.0,
            "gpu_utilization": 0.0,
        }

        try:
            json_resp = json.loads(response.text)
            cpu_raw = json_resp.get("cpu_usage_percent", "0")
            data["cpu_utilization"] = str(float(cpu_raw) / 100)
            gpu_raw = json_resp.get("gpu_utilization_percent", "0")
            data["gpu_utilization"] = str(float(gpu_raw) / 100)
            data["cpu_memory_total_bytes"] = json_resp.get("ram_total_bytes", None)
            data["cpu_memory_used_bytes"] = json_resp.get("ram_used_bytes", "0")
            val_mb = float(json_resp.get("vram_total_mb", "0"))
            data["gpu_memory_total_bytes"] = str(val_mb * 1024 * 1024)
            val_mb = float(json_resp.get("vram_used_mb", "0"))
            data["gpu_memory_used_bytes"] = str(val_mb * 1024 * 1024)
        except json.JSONDecodeError as err:
            raise ValueError(f"Failed to decode JSON from {url}: {err}")

        return data


class ClusterMetricsMonitor:

    def __init__(
        self,
        triton_hosts,
        vllm_hosts,
        flask_hosts,
        metrics_port=8002,
        poll_interval_ms=3000,
    ):
        print("Starting clusters monitor", file=sys.stderr)
        self.poll_interval_ms = poll_interval_ms
        self.stop_signal = mp.Value("H", 0)
        self.monitors = {}
        for info in triton_hosts:
            host = info["hostname"]
            tp = info["servertype"]
            host_full_name = f"{host}_{tp}_{metrics_port}"
            self.monitors[host_full_name] = TritonMonitor(host, tp, metrics_port)
            print("New host:", host_full_name, file=sys.stderr)
        for info in vllm_hosts:
            host = info["hostname"]
            tp = info["servertype"]
            host_full_name = f"{host}_{tp}_{metrics_port}"
            self.monitors[host_full_name] = VLLMMonitor(host, tp, metrics_port)
            print("New host:", host_full_name, file=sys.stderr)
        for info in flask_hosts:
            host = info["hostname"]
            tp = info["servertype"]
            host_full_name = f"{host}_{tp}_8010"
            self.monitors[host_full_name] = VLLMMonitor(host, tp, port=8010)
            print("New host:", host_full_name, file=sys.stderr)

    def get_metrics(self):
        readings = {}
        for host, monitor in self.monitors.items():
            metrics = monitor.get_system_metrics()
            metrics["inference_type"] = monitor.type
            if type(monitor) == TritonMonitor:
                metrics["server_type"] = "triton"
            elif type(monitor) == VLLMMonitor:
                if monitor.port == 8010:
                    metrics["server_type"] = "flask"
                else:
                    metrics["server_type"] = "vllm"
            else:
                metrics["server_type"] = "other"
            readings[host] = metrics
        return readings

    def monitor_clusters(self, batch_time=24):
        sqlite_conn = sqlite.connect(os.environ["SQLITE_DB_PATH"])
        metrics_batch = []
        last_save = time.time()

        monitor_columns = [
            "readings_time",
            "hostname",
            "server_type",
            "inference_type",
            "metrics_dict",
        ]
        sql_insert_cmd = (
            "INSERT INTO metrics ("
            + ", ".join(monitor_columns)
            + ") VALUES ("
            + ", ".join(["?" for _ in monitor_columns])
            + ")"
        )

        while self.stop_signal.value == 0:
            now = time.time()
            current_readings = self.get_metrics()
            for host_full_name, metrics in current_readings.items():
                if not metrics:
                    """print(
                        "Monitor: Host",
                        host_full_name,
                        "returned None",
                        file=sys.stderr,
                    )"""
                    continue
                non_null_values = [
                    v for v in metrics.values() if v not in ["", "None", None, "[]", []]
                ]
                if len(non_null_values) == 0:
                    """print(
                        "Monitor: Host",
                        host_full_name,
                        "returned no metrics",
                        file=sys.stderr,
                    )"""
                    continue
                host = host_full_name.split("_")[0]
                row = {
                    "readings_time": str(now),
                    "hostname": host,
                    "server_type": metrics["server_type"],
                    "inference_type": metrics["inference_type"],
                }
                row["metrics_dict"] = json.dumps(
                    {c: str(metrics.get(c, None)) for c in prometheus_columns},
                    ensure_ascii=False,
                    indent=0,
                )
                # print(row, file=sys.stderr)
                if not '"None"' in row["metrics_dict"]:
                    metrics_batch.append(row)
                else:
                    # print("Not including row due to 'None' values", file=sys.stderr)
                    pass

            if time.time() - last_save >= batch_time:
                if len(metrics_batch) > 0:
                    cursor = sqlite_conn.cursor()
                    # convert dicts to tuples
                    insert_tuples = [
                        tuple([metrics[c] for c in monitor_columns])
                        for metrics in metrics_batch
                    ]
                    # insert into db
                    print(
                        "Monitor: Saving",
                        len(insert_tuples),
                        "readings",
                        file=sys.stderr,
                    )
                    cursor.executemany(sql_insert_cmd, insert_tuples)
                    sqlite_conn.commit()

                    metrics_batch = []
                    last_save = time.time()
            operations_time = time.time() - now
            remaining_time = self.poll_interval_ms - operations_time * 1000
            if remaining_time > 0:
                # print("Monitor: Waiting", remaining_time, "ms", file=sys.stderr)
                time.sleep(remaining_time / 1000.0)

    def stop(self):
        self.stop_signal.value = 1


def start_clusters_monitor(triton_hosts, vllm_hosts, flask_hosts, metrics_port):

    monitor = ClusterMetricsMonitor(
        triton_hosts, vllm_hosts, flask_hosts, metrics_port=metrics_port
    )
    stop_signal = monitor.stop_signal

    # Start process to monitor metrics, without blocking
    monitor_process = mp.Process(target=monitor.monitor_clusters)
    monitor_process.start()
    print("Metrics Monitor started!", file=sys.stderr)

    return monitor, monitor_process, stop_signal


def read_all_prometheus_readings():
    readings = []
    for filename in glob(os.path.join(PROMETHEUS_READINGS_DIR, "*.tsv")):
        with open(filename, "r") as f:
            readings.extend(f.read())
    tsv_lines = []
    for reading in readings:
        lines = reading.split("\n")
        line_with_cols = {
            {prometheus_columns[i]: val for i, val in enumerate(rawline.split("\t"))}
            for rawline in lines
        }
        tsv_lines.extend(line_with_cols)
    return tsv_lines
