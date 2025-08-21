#!/usr/bin/env python3
import argparse
import json
import os
from collections import defaultdict
import pandas as pd
import yaml

def load_dataset(csv_path):
    """Load behaviors dataset into a map: BehaviorID -> (row_index, row_dict)"""
    df = pd.read_csv(csv_path)
    id_to_row = {}
    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        # normalize NaN to None for JSON compatibility
        for k, v in row_dict.items():
            if pd.isna(v):
                row_dict[k] = None
        key = row_dict.get("BehaviorID")
        if key:
            id_to_row[key] = (idx, row_dict)
    return id_to_row

def coerce_turns_needed(x):
    """Ensure turns_needed is int where possible (some models return strings)."""
    if isinstance(x, int):
        return x
    try:
        # e.g., "4" or "4 turns are needed" -> 4
        import re
        m = re.search(r"\d+", str(x))
        if m:
            return int(m.group(0))
    except Exception:
        pass
    return x  # leave as-is if not parseable

def chunk_into_sets(items, per_set=10):
    """
    Convert a flat list of strategy dicts into
    {"Set_1": {"strategy_1": {...}}, "Set_2": {...}}
    """
    sets = {}
    for i in range(0, len(items), per_set):
        set_idx = (i // per_set) + 1
        set_key = f"Set_{set_idx}"
        batch = items[i:i + per_set]
        # Build numbered strategies inside each set
        strategies = {}
        for j, strat in enumerate(batch, start=1):
            # Some progress items might lack fields; keep whatever is there
            s = dict(strat)
            # coerce turns_needed
            if "turns_needed" in s:
                s["turns_needed"] = coerce_turns_needed(s["turns_needed"])
            strategies[f"strategy_{j}"] = s
        sets[set_key] = strategies
    return sets

def rebuild(progress_path, csv_path, out_path):
    # Load progress
    with open(progress_path, "r") as f:
        progress = json.load(f)

    # Load dataset to reconstruct behavior_details
    id_to_row = load_dataset(csv_path)

    final_list = []
    fallback_counter = 1

    for behavior_id, payload in progress.items():
        # Payload can be: {"items": [ {...}, {...} ]} or other shapes.
        # We only support the "items" list case (as in your dump).
        items = []
        if isinstance(payload, dict) and "items" in payload and isinstance(payload["items"], list):
            items = payload["items"]
        elif isinstance(payload, list):
            items = payload
        else:
            # Not recognized shape; skip
            continue

        # Build behavior_details from dataset row
        row_index = None
        behavior_details = None
        if behavior_id in id_to_row:
            row_index, row_dict = id_to_row[behavior_id]
            # Only pick the columns you want in "behavior_details"
            behavior_details = {
                "Behavior": row_dict.get("Behavior"),
                "FunctionalCategory": row_dict.get("FunctionalCategory"),
                "SemanticCategory": row_dict.get("SemanticCategory"),
                "Tags": row_dict.get("Tags"),
                "ContextString": row_dict.get("ContextString"),
                "BehaviorID": row_dict.get("BehaviorID"),
            }
        else:
            # Fallback if we can't find the row in the dataset
            behavior_details = {
                "Behavior": None,
                "FunctionalCategory": None,
                "SemanticCategory": None,
                "Tags": None,
                "ContextString": None,
                "BehaviorID": behavior_id,
            }

        # Create sets "Set_1", "Set_2", ... with 10 per set
        attack_strategies = chunk_into_sets(items, per_set=10)

        final_list.append({
            "behavior_number": (row_index if row_index is not None else fallback_counter),
            "behavior_details": behavior_details,
            "attack_strategies": attack_strategies,
        })
        fallback_counter += 1

    # Sort by behavior_number for consistency
    final_list.sort(key=lambda x: x["behavior_number"])

    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Write final file
    with open(out_path, "w") as f:
        json.dump(final_list, f, indent=4, ensure_ascii=False)

    print(f"[info] Wrote rebuilt attack plans to: {out_path}")
    print(f"[info] Behaviors included: {len(final_list)}")

def main():
    p = argparse.ArgumentParser(description="Rebuild attack_plans.json from progress JSON.")
    p.add_argument("-c", "--config", default="config/config.yaml", help="Path to config YAML (for behavior CSV path)")
    p.add_argument("-p", "--progress", required=True, help="Path to progress JSON (attack_plans_progress.json)")
    p.add_argument("-o", "--out", required=True, help="Path to output attack_plans.json")

    args = p.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    csv_path = cfg["attack_plan_generator"]["behavior_path"]
    rebuild(args.progress, csv_path, args.out)

if __name__ == "__main__":
    main()
