import os
import sys

# Set environment variables FIRST
os.environ.update({
    "MPLBACKEND": "Agg",
    "QT_QPA_PLATFORM": "offscreen", 
    "DISPLAY": "",
    "MPLCONFIGDIR": "/tmp"
})

# Import and configure matplotlib before anything else
import matplotlib
matplotlib.use("Agg", force=True)

# Try to pre-configure matplotlib's internal state
try:
    import matplotlib.pyplot
    matplotlib.pyplot.ioff()
    # Force matplotlib to initialize with Agg backend
    fig = matplotlib.pyplot.figure()
    matplotlib.pyplot.close(fig)
except Exception as e:
    print(f"[WORKER] matplotlib setup warning: {e}", flush=True)

print("[WORKER] mpl version:", matplotlib.__version__, "backend:", matplotlib.get_backend(), flush=True)

# Now import other libraries
import json
import gc
import cv2 as cv
print("[WORKER] cv2 version:", cv.__version__, flush=True)
cv.ocl.setUseOpenCL(False)
cv.setNumThreads(1)

# Finally import your modules
from seasonal_comparison.process_timestamp_batch import seasonalComparer
from seasonal_comparison.general_utils import load_config

def main():
    print("[WORKER] starting", flush=True)
    with open(sys.argv[1], "r") as f:
        args = json.load(f)
    config = load_config(args["config_path"])
    comparer = seasonalComparer(config)
    print(f"[WORKER] {len(args['timestamps'])} timestamps in batch", flush=True)
    for i, t in enumerate(args["timestamps"]):
        try:
            print(f"[WORKER] processing idx={i} ts={t}", flush=True)
            comparer.process_timestamp(i, t)
            print(f"[WORKER] done ts={t}", flush=True)
        except Exception as e:
            print(f"[ERR] timestamp {t} failed: {e}", flush=True)
        finally:
            gc.collect()
    print("[WORKER] batch complete", flush=True)

if __name__ == "__main__":
    main()