import threading
import math
import psutil
import os
import time 
import resource
import time
from datetime import timedelta
import numpy as np

def limit_memory(max_gb: float):
    soft = int(max_gb * (1024**3))  # bytes
    hard = soft
    resource.setrlimit(resource.RLIMIT_AS, (soft, hard))

def _kill_child_processes(grace_sec: float = 2.0):
    """
    Best-effort: send SIGTERM to all children, then SIGKILL after a short grace.
    Works even if the executor is in a bad state.
    """
    try:
        if psutil is None:
            # Fallback: use multiprocessing to terminate known children
            for p in mp.active_children():
                try:
                    p.terminate()
                except Exception:
                    pass
            time.sleep(grace_sec)
            for p in mp.active_children():
                try:
                    p.kill()
                except Exception:
                    pass
            return

        parent = psutil.Process(os.getpid())
        kids = parent.children(recursive=True)
        for k in kids:
            try:
                k.terminate()
            except Exception:
                pass
        gone, alive = psutil.wait_procs(kids, timeout=grace_sec)
        for a in alive:
            try:
                a.kill()
            except Exception:
                pass
    except Exception:
        pass


class ProgressETAThreaded:
    def __init__(self, total: int, alpha: float = 0.2, interval: float = 120.0, label: str = "PROG"):
        """
        total    = total #items
        alpha    = EMA smoothing for sec/item
        interval = seconds between prints
        """
        self.total = max(1, int(total))
        self.alpha = float(alpha)
        self.interval = float(interval)
        self.label = label

        self.start_ts = time.time()
        self.done = 0
        self._ema_sec_per_item = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="ProgressETA", daemon=True)
        self._thread.start()

    def stop(self, force_summary: bool = True):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval * 2)
        if force_summary:
            self._print_snapshot(force=True)

    def hit(self, n: int = 1):
        """Call when an item completes."""
        with self._lock:
            self.done += n
            elapsed = time.time() - self.start_ts
            avg = elapsed / max(1, self.done)
            if self._ema_sec_per_item is None:
                self._ema_sec_per_item = avg
            else:
                self._ema_sec_per_item = self.alpha * avg + (1 - self.alpha) * self._ema_sec_per_item

    # ---- internals ----
    def _run(self):
        # periodic printer
        while not self._stop.wait(self.interval):
            self._print_snapshot(force=False)

    def _print_snapshot(self, force: bool = False):
        with self._lock:
            done = self.done
            ema = self._ema_sec_per_item

        now = time.time()
        elapsed = now - self.start_ts
        if ema is None:
            # no completions yet -> print a minimal line
            msg = (f"[{self.label}] {done}/{self.total} (  0.0%) | "
                   f"elapsed={self._fmt_td(elapsed)} eta=?:??:??")
        else:
            remain = max(0, self.total - done)
            eta_sec = remain * ema
            pct = 100.0 * done / self.total
            msg = (f"[{self.label}] {done}/{self.total} ({pct:5.1f}%) | "
                   f"ema/item={ema:0.2f}s | elapsed={self._fmt_td(elapsed)} eta={self._fmt_td(eta_sec)}")

        print(msg, flush=True)

    @staticmethod
    def _fmt_td(sec: float) -> str:
        if sec < 0 or not np.isfinite(sec):
            return "?:??:??"
        return str(timedelta(seconds=int(sec)))


class ResourceMonitor:
    """
    Periodically logs resource usage of the parent process and all child processes (e.g., ProcessPool workers).
    Prints: #procs, ~#workers, total threads, total RSS, proc mem %, system mem %, total CPU %.
    """
    def __init__(self, interval_sec: float = 5.0, label: str = "MON"):
        self.interval = max(0.5, float(interval_sec))
        self.label = label
        self._stop = threading.Event()
        self._thread = None
        self._proc = psutil.Process(os.getpid()) if psutil else None
        # Track max seen values for a quick summary at the end
        self._max = {
            "rss": 0, "cpu": 0.0, "mem_pct": 0.0, "threads": 0, "procs": 0
        }

    def start(self):
        if not psutil:
            print("[WARN] psutil not available; resource monitoring disabled.", flush=True)
            return
        # Warm up CPU percent counters
        try:
            for p in [self._proc] + self._proc.children(recursive=True):
                _ = p.cpu_percent(None)
        except Exception:
            pass

        self._thread = threading.Thread(target=self._run, name="ResourceMonitor", daemon=True)
        self._thread.start()

    def stop(self):
        if not psutil or not self._thread:
            return
        self._stop.set()
        self._thread.join(timeout=self.interval * 2)
        # Final summary
        print(
            f"[{self.label}-SUMMARY] max_cpu%={self._max['cpu']:.1f} "
            f"max_mem%={self._max['mem_pct']:.1f} max_RSS={self._fmt_bytes(self._max['rss'])} "
            f"max_threads={self._max['threads']} max_procs={self._max['procs']}",
            flush=True
        )

    def _run(self):
        while not self._stop.is_set():
            try:
                self._snapshot()
            except Exception as e:
                print(f"[{self.label}] monitor error: {e}", flush=True)
            self._stop.wait(self.interval)

    def _snapshot(self):
        # Re-discover children each iteration (pool can grow/shrink)
        procs = [self._proc] + self._proc.children(recursive=True)
        # Prime CPU counters (interval-less)
        for p in list(procs):
            try:
                _ = p.cpu_percent(None)
            except Exception:
                pass

        # Sleep for the interval to measure CPU %
        time.sleep(self.interval)

        rss = 0
        mem_pct_sum = 0.0
        cpu_pct_sum = 0.0
        threads = 0
        alive_procs = 0

        for p in list(procs):
            try:
                if not p.is_running():
                    continue
                with p.oneshot():
                    mi = p.memory_info()
                    rss += getattr(mi, "rss", 0)
                    mem_pct_sum += p.memory_percent()
                    cpu_pct_sum += p.cpu_percent(None)
                    threads += p.num_threads()
                    alive_procs += 1
            except Exception:
                continue

        sys_mem_pct = psutil.virtual_memory().percent if psutil else float("nan")
        workers = max(0, alive_procs - 1)  # minus parent

        # Track maxima
        self._max["rss"] = max(self._max["rss"], rss)
        self._max["mem_pct"] = max(self._max["mem_pct"], mem_pct_sum)
        self._max["cpu"] = max(self._max["cpu"], cpu_pct_sum)
        self._max["threads"] = max(self._max["threads"], threads)
        self._max["procs"] = max(self._max["procs"], alive_procs)

        print(
            f"[{self.label}] procs={alive_procs} workers≈{workers} threads={threads} "
            f"RSS={self._fmt_bytes(rss)} proc_mem%={mem_pct_sum:.1f} "
            f"sys_mem%={sys_mem_pct:.1f} cpu%={cpu_pct_sum:.1f}",
            flush=True
        )

    @staticmethod
    def _fmt_bytes(b):
        # Human-friendly bytes
        if b <= 0: return "0 B"
        units = ["B", "KB", "MB", "GB", "TB"]
        i = min(len(units) - 1, int(math.log(b, 1024)))
        return f"{b / (1024 ** i):.2f} {units[i]}"
