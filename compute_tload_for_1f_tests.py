#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import sys
import yaml
from file_access import load_metadata
from tnoise_step2 import extract_temperatures


def main(args):
    if len(args) < 2:
        print(f"Usage: {args[0]} YAML_FILE URL1...")
        sys.exit(1)

    with open(args[1], "rt") as f:
        db = yaml.load(f)

    for cur_arg in args[2:]:
        try:
            metadata = load_metadata(cur_arg)
            temp_a, temp_b = extract_temperatures(metadata)
            temp_a, temp_b, temp_ave = temp_a[0], temp_b[0], (temp_a[0] + temp_b[0]) / 2
        except IndexError:
            continue

        for cur_id in range(len(db)):
            if db[cur_id]["id"] != metadata["polarimeter_number"]:
                continue

            db[cur_id]["spectrum"]["load_a_temperature_k"] = float(temp_a)
            db[cur_id]["spectrum"]["load_b_temperature_k"] = float(temp_b)
            db[cur_id]["spectrum"]["load_average_temperature_k"] = float(temp_ave)

            break

    print(yaml.dump(db))


if __name__ == "__main__":
    main(sys.argv)
