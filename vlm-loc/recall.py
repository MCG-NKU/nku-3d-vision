import json
import math
import argparse


PX_PER_50M = 224.0
METERS_PER_PX = 50.0 / PX_PER_50M  # 224px -> 50m


def safe_json_loads(x):
    if x is None:
        return None
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        x = x.strip()
        if not x:
            return None
        try:
            return json.loads(x)
        except json.JSONDecodeError:
            return None
    return None


def extract_point(obj):
    if not obj or "point_2d" not in obj:
        return None
    p = obj["point_2d"]
    if not isinstance(p, (list, tuple)) or len(p) != 2:
        return None
    try:
        return float(p[0]), float(p[1])
    except:
        return None


def dist_px(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def main():
    parser = argparse.ArgumentParser()
    # Note: Modify the result path inferred by test.sh 
    parser.add_argument("--jsonl", default="/home/data_sata/vlmloc/ms-swift_qwen3/full_partial_results/20251101-211525.jsonl")
    parser.add_argument("--skip_invalid", action="store_true")
    args = parser.parse_args()

    total = 0
    valid = 0
    invalid = 0
    hits_5 = hits_10 = hits_15 = 0
    dists_m = []

    with open(args.jsonl, "r") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total += 1

            try:
                row = json.loads(line)
            except:
                invalid += 1
                continue

            pred_obj = safe_json_loads(row.get("response"))
            gt_obj = safe_json_loads(row.get("labels"))

            pred_pt = extract_point(pred_obj)
            gt_pt = extract_point(gt_obj)

            if pred_pt is None or gt_pt is None:
                invalid += 1
                if args.skip_invalid:
                    continue
                else:
                    continue

            valid += 1
            dpx = dist_px(pred_pt, gt_pt)
            dm = dpx * METERS_PER_PX
            dists_m.append(dm)

            if dm <= 5: hits_5 += 1
            if dm <= 10: hits_10 += 1
            if dm <= 15: hits_15 += 1

    denom = valid if args.skip_invalid else total

    print("Total:", total)
    print("Valid:", valid)
    print("Invalid:", invalid)
    print("m/px:", METERS_PER_PX)
    print(f"R@5m  = {hits_5/denom:.4f}")
    print(f"R@10m = {hits_10/denom:.4f}")
    print(f"R@15m = {hits_15/denom:.4f}")


if __name__ == "__main__":
    main()
