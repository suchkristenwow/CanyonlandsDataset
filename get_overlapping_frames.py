#!/usr/bin/env python3
import os, json, tempfile, subprocess, sys, time, select
from pathlib import Path
import numpy as np
from seasonal_comparison.general_utils import log_mem, parse_args, load_config

def run_worker_with_logs(json_path, timeout_sec=600):
    from pathlib import Path
    import tempfile, getpass

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.update({
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "OPENCV_OPENCL_RUNTIME": "disabled",
        "MPLBACKEND": "Agg",
        "QT_QPA_PLATFORM": "offscreen",
    })

    # ensure your package is importable (safe even if you already pip-installed it)
    repo_root = Path(__file__).resolve().parent
    src_path = str((repo_root / "src").resolve())
    env["PYTHONPATH"] = src_path + (":" + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

    # give Matplotlib a writable config dir to avoid cache/font-lock issues
    mpldir = os.path.join(tempfile.gettempdir(), f"mpl_{getpass.getuser()}_{os.getpid()}")
    os.makedirs(mpldir, exist_ok=True)
    env["MPLCONFIGDIR"] = mpldir

    cmd = [sys.executable, "-u", "-m", "seasonal_comparison.timestamp_batch_worker", json_path]
    print("[LAUNCH]", " ".join(cmd), flush=True)

    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        bufsize=1,
    )

    start = time.time()
    stdout_buf, stderr_buf = [], []

    while True:
        rlist, _, _ = select.select([p.stdout, p.stderr], [], [], 0.25)
        for pipe in rlist:
            line = pipe.readline()
            if not line:
                continue
            if pipe is p.stdout:
                stdout_buf.append(line); print(line, end="", flush=True)
            else:
                stderr_buf.append(line); print(line, end="", file=sys.stderr, flush=True)

        if p.poll() is not None:
            break
        if time.time() - start > timeout_sec:
            print("[WARN] Worker timed out; terminating...", flush=True)
            p.kill()
            try: p.wait(5)
            except Exception: pass
            return 124, "".join(stdout_buf), "".join(stderr_buf)

    rc = p.returncode
    if rc != 0:
        print(f"[ERROR] Worker exited with code {rc}", flush=True)
    return rc, "".join(stdout_buf), "".join(stderr_buf)

def main():
    args = parse_args()
    config = load_config(args.config)
    paths = config["paths"]

    ts_path = os.path.join(paths["may_results"], "frustrum_corners.csv")
    timestamps = np.loadtxt(ts_path, delimiter=",", skiprows=1, usecols=[0])

    chunk_size = 3

    for i in range(0, len(timestamps), chunk_size):
        batch = timestamps[i:i+chunk_size]
        print(f"[INFO] Processing batch {i//chunk_size + 1}")
        log_mem()

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            data_path = f.name
            json.dump({"config_path": args.config, "timestamps": batch.tolist()}, f)

        rc, out, err = run_worker_with_logs(data_path)
        try:
            os.remove(data_path)
        except OSError:
            pass

        print("[INFO] End of batch process")
        if rc != 0:
            # optional: break or continue depending on your tolerance
            print("[INFO] Continuing despite worker error.")

if __name__ == "__main__":
    main()
