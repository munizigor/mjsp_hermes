import atexit
import fcntl
import os
import shutil
import signal
import subprocess
import sys
import threading
from contextlib import contextmanager
from shutil import copy as cp
from typing import Dict, Optional


def set_output_permissions(output_dir: str):
    """
    Recursively grants all users write access to directories (and the execute bit to enter them),
    and read-only access to files. Assumes execution as sudo.
    """
    if not output_dir or not os.path.exists(output_dir):
        print(f"⚠️ Directory not found, skipping permissions: {output_dir}", flush=True)
        return

    if os.geteuid() != 0:
        print(
            "⚠️ Warning: Script is not running as root. os.chmod may fail on files you don't own.",
            flush=True,
        )

    dir_mode = 0o777
    file_mode = 0o644

    try:
        os.chmod(output_dir, dir_mode)

        for root, dirs, files in os.walk(output_dir):
            for d in dirs:
                dir_path = os.path.join(root, d)
                os.chmod(dir_path, dir_mode)

            for f in files:
                file_path = os.path.join(root, f)
                os.chmod(file_path, file_mode)

        print(f"✅ Permissions successfully applied to {output_dir}", flush=True)

    except Exception as e:
        print(f"❌ Failed to set permissions in {output_dir}: {e}", flush=True)


class DockerController:
    """
    Thread-safe and multiprocessing-safe controller for managing the Docker Compose
    lifecycle, container log streaming, and test output directory permissions.
    """

    def __init__(
        self,
        proj_dir: str,
        output_dir: Optional[str] = None,
        env_vals: Optional[Dict[str, str]] = None,
        register_signals: bool = True,
        register_atexit: bool = True,
    ):
        self.proj_dir = os.path.abspath(proj_dir)
        self._output_dir = os.path.abspath(output_dir) if output_dir else None
        self.env_vals = dict(env_vals) if env_vals is not None else self._load_dotenv()

        # Re-entrant synchronization state
        self._lock = threading.RLock()
        self._lock_depth = 0
        self._lock_file = None
        self._owner_pid = os.getpid()

        self._log_process: Optional[subprocess.Popen] = None
        self._log_file = None
        self._is_running = False

        self._lock_file_path = os.path.join(self.proj_dir, ".docker_controller.lock")
        self._state_file_path = os.path.join(
            self.proj_dir, ".docker_controller_output_dir"
        )

        if self._output_dir:
            self._write_shared_output_dir(self._output_dir)

        self.register_cleanup_handlers(
            register_signals=register_signals,
            register_atexit=register_atexit,
        )

    # -------------------------------------------------------------------------
    # Multiprocessing Pickle Support & Re-entrant Cross-Process Synchronization
    # -------------------------------------------------------------------------
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_lock"] = None
        state["_lock_file"] = None
        state["_lock_depth"] = 0
        state["_log_process"] = None
        state["_log_file"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.RLock()
        self._lock_file = None
        self._lock_depth = 0

    @contextmanager
    def _synchronized(self):
        """
        Re-entrant cross-process and cross-thread lock.
        Guarantees that nested calls within the same process do not deadlock on fcntl.flock.
        """
        with self._lock:
            if self._lock_depth == 0:
                os.makedirs(self.proj_dir, exist_ok=True)
                self._lock_file = open(self._lock_file_path, "a")
                fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX)
            self._lock_depth += 1
            try:
                yield
            finally:
                self._lock_depth -= 1
                if self._lock_depth == 0:
                    try:
                        fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
                    finally:
                        self._lock_file.close()
                        self._lock_file = None

    def _write_shared_output_dir(self, path: Optional[str]):
        try:
            if path:
                with open(self._state_file_path, "w") as f:
                    f.write(path)
            elif os.path.exists(self._state_file_path):
                os.remove(self._state_file_path)
        except OSError:
            pass

    def _read_shared_output_dir(self) -> Optional[str]:
        try:
            if os.path.exists(self._state_file_path):
                with open(self._state_file_path, "r") as f:
                    content = f.read().strip()
                    if content:
                        return content
        except OSError:
            pass
        return self._output_dir

    def _load_dotenv(self) -> Dict[str, str]:
        env_path = os.path.join(self.proj_dir, ".env")
        if not os.path.exists(env_path):
            return {}
        with open(env_path, "r") as f:
            return {
                line.split("=", 1)[0]: line.split("=", 1)[1].rstrip("\n")
                for line in f.read().splitlines()
                if "=" in line and not line.lstrip().startswith("#")
            }

    # -------------------------------------------------------------------------
    # Dynamic output_dir Management
    # -------------------------------------------------------------------------
    @property
    def output_dir(self) -> Optional[str]:
        with self._lock:
            return self._read_shared_output_dir()

    @output_dir.setter
    def output_dir(self, new_output_dir: Optional[str]):
        self.set_output_dir(new_output_dir)

    def set_output_dir(self, new_output_dir: Optional[str]):
        """Thread-safe and process-safe setter for the active test output directory."""
        with self._synchronized():
            resolved = os.path.abspath(new_output_dir) if new_output_dir else None
            self._output_dir = resolved
            self._write_shared_output_dir(resolved)

    # -------------------------------------------------------------------------
    # Signal & Exit Handlers
    # -------------------------------------------------------------------------
    def register_cleanup_handlers(
        self, register_signals: bool = True, register_atexit: bool = True
    ):
        if register_atexit:
            atexit.register(self._atexit_cleanup)

        if (
            register_signals
            and threading.current_thread() is threading.main_thread()
        ):
            signal.signal(signal.SIGINT, self.handle_interrupt)
            signal.signal(signal.SIGTERM, self.handle_interrupt)

    def _atexit_cleanup(self):
        # Prevent forked worker processes from tearing down Docker when they exit
        if os.getpid() != self._owner_pid:
            return
        if self._is_running:
            self.docker_stop()

    def handle_interrupt(self, signum, frame):
        """Catches system signals (like CTRL+C) and forces a graceful exit."""
        if os.getpid() != self._owner_pid:
            sys.exit(1)

        print(
            f"\n⚠️ Received interrupt signal ({signum}). Exiting gracefully...",
            flush=True,
        )
        self.docker_stop()
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Docker Lifecycle Operations
    # -------------------------------------------------------------------------
    def _stop_log_streamer(self):
        """Cleanly stops the background `docker compose logs -f` process and closes the file."""
        if self._log_process is not None:
            try:
                if self._log_process.poll() is None:
                    self._log_process.terminate()
                    self._log_process.wait(timeout=5)
            except Exception:
                try:
                    self._log_process.kill()
                except Exception:
                    pass
            finally:
                self._log_process = None

        if self._log_file is not None:
            try:
                self._log_file.flush()
                self._log_file.close()
            except Exception:
                pass
            finally:
                self._log_file = None

    def set_permissions(self, target_output_dir: Optional[str] = None):
        """Applies permissions to the specified or currently active output_dir."""
        with self._synchronized():
            out_dir = target_output_dir or self._read_shared_output_dir()
            if out_dir and os.path.exists(out_dir):
                set_output_permissions(out_dir)

    def docker_stop(
        self,
        output_dir: Optional[str] = None,
        apply_permissions: bool = True,
    ):
        """Tears down the Docker Compose stack and optionally sets output permissions."""
        with self._synchronized():
            if output_dir is not None:
                self.set_output_dir(output_dir)

            self._stop_log_streamer()

            print("🧹 Tearing down Docker Compose stack...", flush=True)
            subprocess.run(
                ["docker", "compose", "down"],
                cwd=self.proj_dir,
                check=False,
            )
            self._is_running = False
            print("🏁 Cleanup complete.", flush=True)

            if apply_permissions:
                target_dir = self._read_shared_output_dir()
                if target_dir and os.path.exists(target_dir):
                    set_output_permissions(target_dir)

    def docker_run(self):
        """Ensures previous containers are stopped, cleans cache/DB, and starts Docker Compose."""
        with self._synchronized():
            print("Making sure docker is not running yet...", flush=True)
            self.docker_stop(apply_permissions=False)

            print("🚀 Starting Docker Compose stack...", flush=True)
            success = False
            try:
                run_env = os.environ.copy()
                run_env.update(self.env_vals)

                out_log_path = os.path.join(self.proj_dir, "start_stdout.log")
                err_log_path = os.path.join(self.proj_dir, "start_stderr.log")

                # Remove cached files/directories before starting
                to_remove = [
                    os.path.join(self.proj_dir, "datasets", "naturezas_cache_vllm.json"),
                ]
                sql_db_rel = run_env.get("LOCAL_SQL_DB_PATH")
                if sql_db_rel:
                    to_remove.append(os.path.join(self.proj_dir, sql_db_rel))

                for path in to_remove:
                    if os.path.exists(path):
                        if os.path.isdir(path) and not os.path.islink(path):
                            shutil.rmtree(path, ignore_errors=True)
                        else:
                            try:
                                os.remove(path)
                            except OSError:
                                pass

                cmd_vec = ["docker", "compose", "up", "-d", "--wait"]
                print("Running command:", " ".join(cmd_vec), flush=True)
                with open(out_log_path, "w") as out_log, open(
                    err_log_path, "w"
                ) as err_log:
                    subprocess.run(
                        cmd_vec,
                        check=True,
                        text=True,
                        cwd=self.proj_dir,
                        env=run_env,
                        stdout=out_log,
                        stderr=err_log,
                    )
                print("✅ All containers are up and healthy!", flush=True)

                # Stream container logs to file in the background
                log_path = os.path.join(self.proj_dir, "containers.log")
                self._log_file = open(log_path, "w")

                print(f"📝 Streaming container application logs to {log_path}...", flush=True)
                self._log_process = subprocess.Popen(
                    ["docker", "compose", "logs", "-f"],
                    cwd=self.proj_dir,
                    env=run_env,
                    stdout=self._log_file,
                    stderr=subprocess.STDOUT,
                )

                self._is_running = True
                success = True
            except subprocess.CalledProcessError as e:
                print(
                    f"❌ Docker Compose failed to start or health checks timed out: {e}",
                    flush=True,
                )
            except Exception as e:
                print(f"❌ An error occurred during request execution: {e}", flush=True)
            finally:
                if not success:
                    self.docker_stop(apply_permissions=True)
                    sys.exit(1)

    def start_docker(
        self,
        config_json_path: str,
        output_dir: Optional[str] = None,
    ):
        """Copies config.json into proj_dir, updates output_dir if provided, and starts Docker."""
        with self._synchronized():
            if output_dir is not None:
                self.set_output_dir(output_dir)
            cp(config_json_path, os.path.join(self.proj_dir, "config.json"))
            self.docker_run()

    # Aliases
    start = start_docker
    stop = docker_stop
    run = docker_run