import sys
import json
from seasonal_comparison.process_timestamp_batch import seasonalComparer
from seasonal_comparison.general_utils import load_config 

def main():
    with open(sys.argv[1], "r") as f:
        args = json.load(f)

    config = load_config(args["config_path"])
    comparer = seasonalComparer(config)

    for i, t in enumerate(args["timestamps"]):
        comparer.process_timestamp(i, t)

if __name__ == "__main__":
    main()
