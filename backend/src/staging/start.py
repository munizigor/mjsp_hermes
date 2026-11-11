import os
import signal
import sys
from time import sleep

env_file_path = "/.env" if os.path.exists("/.env") else ".env"
if os.path.exists(env_file_path):
    with open(env_file_path, "r") as f:
        env_vals = {
            rawline.split("=", 1)[0]: rawline.split("=", 1)[1].rstrip("\r\n")
            for rawline in f.read().split("\n")
            if "=" in rawline and not rawline.lstrip().startswith("#")
        }
    for k, v in env_vals.items():
        os.environ[k] = v

from monitors.asterisk_monitor_dependent import start_asterisk_monitor


def main():
    monitor = None
    monitor_process = None
    stop_signal = None

    try:
        use_asterisk = os.environ.get("USE_ASTERISK", "FALSE") == "TRUE"
        if use_asterisk:
            print("Starting Asterisk Monitor")
            monitor, monitor_process, stop_signal = start_asterisk_monitor()

        print("Staging service started successfully!")
    except Exception as err:
        print("Error at staging startup")
        print(err)
        quit(1)

    def handle_stop(signum, frame):
        if stop_signal is not None and monitor_process is not None:
            print("Stopping Asterisk Monitor")
            stop_signal.value = 1
            monitor_process.join()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)

    try:
        while True:
            sleep(30)
    finally:
        if stop_signal is not None and monitor_process is not None:
            print("Stopping Asterisk Monitor")
            stop_signal.value = 1
            monitor_process.join()


if __name__ == "__main__":
    main()
