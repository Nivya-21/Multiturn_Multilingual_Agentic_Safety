import argparse
import concurrent.futures
import json
import logging
import os
from datetime import datetime
import re
from pprint import pformat
import numpy as np
import pandas as pd
import tqdm
import yaml
from tenacity import retry, stop_after_attempt, wait_fixed
from utils.helpers import extract_all_json_objects, normalize_strategies, merge_and_save

from agents.base_agent import BaseAgent


def setup_logging(output_dir):
    """Setup logging to both file and console with ANSI code handling"""

    class NoColorFormatter(logging.Formatter):
        def format(self, record):
            import re

            record.msg = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", str(record.msg))
            return super().format(record)

    file_handler = logging.FileHandler(os.path.join(output_dir, "generation_log.txt"))
    file_handler.setFormatter(
        NoColorFormatter("%(asctime)s - %(threadName)s - %(levelname)s - %(message)s")
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])


def load_datasets(csv_path, number_of_behaviors=15):
    """Load and filter dataset"""
    np.random.seed(27)
    df = pd.read_csv(csv_path)
    filtered_df = df[df["FunctionalCategory"] == "standard"]
    return filtered_df.sample(n=number_of_behaviors)

def extract_json_from_output(raw_content: str) -> dict:
    """
    Extracts the first JSON object from a string, even if it's inside a code block.
    Returns a Python dict. Raises ValueError if no JSON found.
    """
    # Find all {...} blocks (including nested ones)
    matches = re.findall(r'```json\s*([\s\S]*?)```', raw_content)
    if matches:
        json_str = matches[0]
    else:
        # Try to find first { ... } block
        matches = re.findall(r'({[\s\S]+})', raw_content)
        if matches:
            json_str = matches[0]
        else:
            raise ValueError("No JSON object found in model output.")
    # Try to parse
    try:
        return json.loads(json_str)
    except Exception as e:
        raise ValueError(f"JSON parsing failed: {e}\nContent: {json_str}")

def load_and_format_prompts(behavior, set_number, previous_responses=None,attack_language="auto"):
    """Load and format prompts based on set number"""
    with open("config/prompts/plan_generation_prompts.yaml", "r") as f:
        prompts = yaml.safe_load(f)

    system_prompt = prompts["prompts"]["system"]["messages"][0]["content"]
    system_prompt = system_prompt.replace("{attack_language}", attack_language)

    if set_number == 1:
        user_prompt = prompts["prompts"]["user_message1"]["messages"][0]["content"]
        formatted_user_prompt = (
            user_prompt
            .replace("{target_behavior}", behavior)
            .replace("{attack_language}", attack_language)
        )
    else:
        user_prompt = prompts["prompts"]["user_message2"]["messages"][0]["content"]
        formatted_user_prompt = (
            user_prompt
            .replace("{target_behavior}", behavior)
            .replace("{attack_language}", attack_language)
        )

        strategies_text = ""
        for set_name, response in previous_responses.items():
            strategies_text += f"\n{set_name}:\n{response}\n"
        formatted_user_prompt = formatted_user_prompt.replace(
            "{previously_generated_strategies}", strategies_text
        )

    return system_prompt, formatted_user_prompt


def create_output_directory(base_output_dir):
    """Create timestamped output directory"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(base_output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


@retry(stop=stop_after_attempt(5), wait=wait_fixed(1))
def generate_strategies(agent, messages, set_num, temperature, *, behavior_id, progress_dir):
    """Call model, salvage any valid strategies, persist immediately, return what we kept."""
    raw = agent.call_api(
        messages=messages,
        temperature=temperature,
        response_format={"type": "json_object"},
    )

    PROGRESS_PATH = os.path.join(progress_dir, "attack_plans_progress.json")
    good_total = 0
    combined = {}

    # --- Robust handling of different possible return types ---
    if raw is None:
        # Raise to trigger tenacity retry
        raise ValueError("Model returned None for this request")

    if isinstance(raw, dict):
        # Some providers (or JSON mode) return a parsed dict payload
        obj_iter = [raw]
    else:
        # Ensure string
        if not isinstance(raw, str):
            # try to coerce to string if it's something else
            raw = json.dumps(raw, ensure_ascii=False)
        if not raw.strip():
            raise ValueError("Empty string returned from model")
        # extract_all_json_objects() expects a string and returns a list of dicts
        obj_iter = extract_all_json_objects(raw)

    # --- Process each extracted JSON object ---
    for obj in obj_iter:
        picked = normalize_strategies(obj)  # keep only strategy_* and schema-valid
        if not picked:
            continue
        added = merge_and_save(PROGRESS_PATH, behavior_id, picked)  # atomic merge
        good_total += added
        combined.update(picked)

    if good_total == 0:
        logging.warning("[warn] no valid strategies salvaged from this response; skipping.")
    else:
        logging.info(f"[info] saved {good_total} strategy(ies) for behavior {behavior_id}.")

    logging.info(f"\nSet {set_num} Generated Strategies (kept {len(combined)}):")
    logging.info(json.dumps(combined, indent=2, ensure_ascii=False))

    return combined

    
def process_single_behavior(i, row, agent, temperature, progress_dir, num_sets=5):
    behavior = row["Behavior"]
    behavior_id = row["BehaviorID"]
    logging.info(f"\n{'='*50}")
    logging.info(f"Processing Behavior {i} (ID: {behavior_id}):")
    logging.info(f"Behavior: {behavior}")
    logging.info(f"{'='*50}")

    all_messages = []

    # Initialize behavior data
    all_responses = {}
    behavior_details = {
        k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()
    }
    behavior_data = {
        "behavior_number": i,
        "behavior_details": behavior_details,
        "attack_strategies": all_responses,
    }

    # Generate strategies for each set
    for set_num in range(1, num_sets + 1):
        logging.info(f"\nGenerating Set {set_num}:")

        system_prompt, formatted_user_prompt = load_and_format_prompts(
        behavior=behavior,
        set_number=set_num,
        previous_responses=all_responses,
        attack_language=agent.config.get("attack_language", "auto")  # from config
    )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": formatted_user_prompt},
        ]

        logging.info(f"Messages for set {set_num}")
        logging.info(pformat(messages, indent=2))  # instead of pprint(...)


        message_data = {
            "behavior_index": i,
            "behavior": behavior,
            "set_number": set_num,
            "messages": messages,
        }
        all_messages.append(message_data)

        response = generate_strategies(
            agent=agent,
            messages=messages,
            set_num=set_num,
            temperature=temperature,
            behavior_id=behavior_id,
            progress_dir=progress_dir,   # pass from config (see main)
        )

        all_responses[f"Set_{set_num}"] = response

    return behavior_data, all_messages


def main():
    args = argparse.ArgumentParser(
        description="Generates multi-turn jailbreak attack strategies for X-Teaming."
    )
    args.add_argument(
        "-c", "--config", action="store", type=str, default="./config/config.yaml"
    )
    args.add_argument(
        "-L", "--language", default=None,
        help="Language for conversation snippets in the plan (e.g. 'Hindi', 'French', 'English', or 'auto')."
    )
    parsed_args = args.parse_args()

    config_path = parsed_args.config

    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # # Setup
    # output_dir = create_output_directory(
    #     config["attack_plan_generator"]["attack_plan_generation_dir"]
    # )
    apg_cfg = config["attack_plan_generator"]
    if parsed_args.language:
        apg_cfg["attack_language"] = parsed_args.language
    else:
        apg_cfg.setdefault("attack_language", "auto")
    base_out_dir = apg_cfg["attack_plan_generation_dir"]
    output_dir = create_output_directory(base_out_dir)
    setup_logging(output_dir)
    agent = BaseAgent(config["attack_plan_generator"])
    # NEW: progress_dir from config; fall back to base_out_dir/progress
    progress_dir = apg_cfg.get("progress_dir",
                               os.path.join(base_out_dir, "progress"))
        # make sure dirs exist
    os.makedirs(base_out_dir, exist_ok=True)
    os.makedirs(progress_dir, exist_ok=True)

    df = load_datasets(
         apg_cfg["behavior_path"],
         apg_cfg["num_behaviors"],
    )

    all_behaviors_data = []
    all_messages = []

    all_params = []
    for i, row in df.iterrows():
        all_params.append(
            {
                "i": i,
                "row": row,
                "agent": agent,
                "temperature": apg_cfg["temperature"],
                "progress_dir": progress_dir,   # <- pass into worker,
            }
        )

    # Process each behavior
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(process_single_behavior, **args): args
            for args in all_params
        }
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            behavior_result, messages_result = future.result()
            all_behaviors_data.append(behavior_result)
            all_messages += messages_result
            # Save results
            with open(os.path.join(output_dir, "attack_plans.json"), "w") as f:
                json.dump(all_behaviors_data, f, indent=4, ensure_ascii=False)

            with open(os.path.join(output_dir, "all_messages.json"), "w") as f:
                json.dump(all_messages, f, indent=4, ensure_ascii=False)
    logging.info("Finished")


if __name__ == "__main__":
    main()