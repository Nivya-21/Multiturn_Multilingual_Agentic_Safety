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

    # Ensure the directory exists before setting up the file handler
    os.makedirs(output_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(output_dir, "generation_log.txt"))
    file_handler.setFormatter(
        NoColorFormatter("%(asctime)s - %(threadName)s - %(levelname)s - %(message)s")
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    # Remove existing handlers to avoid duplicate logs in case of re-setup
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])


def parse_functions_column(val):
    """Return list[str] of exact tool names."""
    if val is None:
        return []
    s = str(val).strip()
    # Try JSON array first
    try:
        obj = json.loads(s)
        if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
            return obj
    except Exception:
        pass
    # Fallback: split on commas
    parts = [p.strip() for p in re.split(r'\s*,\s*', s) if p.strip()]
    return parts
    

def load_datasets(csv_path, number_of_behaviors=15):
    """Load and filter dataset"""
    np.random.seed(27)
    df = pd.read_csv(csv_path)
    filtered_df = df[df["FunctionalCategory"] == "standard"]
    # If number_of_behaviors is larger than the df, just return the whole df
    if number_of_behaviors > len(filtered_df):
        return filtered_df
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


def load_and_format_prompts(behavior, set_number, previous_responses=None,
                            attack_language="auto", target_functions=None):
    with open("config/prompts/plan_generation_prompts.yaml", "r") as f:
        prompts = yaml.safe_load(f)

    system_prompt = prompts["prompts"]["system"]["messages"][0]["content"]
    system_prompt = system_prompt.replace("{attack_language}", attack_language)

    # Prepare tool placeholders
    target_functions = target_functions or []
    tools_csv = ", ".join(target_functions)
    tools_json = json.dumps(target_functions, ensure_ascii=False)

    if set_number == 1:
        user_prompt = prompts["prompts"]["user_message1"]["messages"][0]["content"]
        formatted_user_prompt = (
            user_prompt
            .replace("{target_behavior}", behavior)
            .replace("{attack_language}", attack_language)
            .replace("{target_functions_comma_separated}", tools_csv)
            .replace("{target_tools_json}", tools_json)
        )
    else:
        user_prompt = prompts["prompts"]["user_message2"]["messages"][0]["content"]
        formatted_user_prompt = (
            user_prompt
            .replace("{target_behavior}", behavior)
            .replace("{attack_language}", attack_language)
            .replace("{target_functions_comma_separated}", tools_csv)
            .replace("{target_tools_json}", tools_json)
        )

        strategies_text = ""
        for set_name, response in (previous_responses or {}).items():
            strategies_text += f"\n{set_name}:\n{json.dumps(response, ensure_ascii=False)}\n"
        formatted_user_prompt = formatted_user_prompt.replace(
            "{previously_generated_strategies}", strategies_text or "{}"
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
        logging.warning(f"[warn] no valid strategies salvaged for behavior {behavior_id} in set {set_num}; response may have been a refusal or malformed.")
    else:
        logging.info(f"[info] saved {good_total} strategy(ies) for behavior {behavior_id}.")

    logging.info(f"\nSet {set_num} Generated Strategies (kept {len(combined)}):")
    logging.info(json.dumps(combined, indent=2, ensure_ascii=False))

    return combined


def process_single_behavior(i, row, agent, temperature, progress_dir, num_sets=5, default_grading_function: str = ""):
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

    # pull from CSV, fall back to config-supplied default
    csv_gf = (
        behavior_details.get("grading_function")
        or behavior_details.get("GradingFunction")
        or behavior_details.get("gradingFunction")
        or default_grading_function
    )

    behavior_data = {
        "behavior_number": i,
        "behavior_details": behavior_details,
        "attack_strategies": all_responses,
    }
    functions_raw = row.get("Functions") if "Functions" in row else None
    target_functions = parse_functions_column(functions_raw)

    # store final gf back into behavior_details
    behavior_details["grading_function"] = csv_gf

    # Generate strategies for each set
    for set_num in range(1, num_sets + 1):
        logging.info(f"\nGenerating Set {set_num}:")

        system_prompt, formatted_user_prompt = load_and_format_prompts(
            behavior=behavior,
            set_number=set_num,
            previous_responses=all_responses,
            target_functions=target_functions,
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
            "behavior_id": behavior_id, # Add behavior_id for easier filtering
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
            progress_dir=progress_dir,
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
    args.add_argument(
        "-r", "--resume", action="store_true",
        help="Resume the latest unfinished attack plan generation run."
    )
    # ===== START: UPDATED ARGUMENT DEFINITION =====
    args.add_argument(
        "--resume-empty", action="store_true",
        help="When resuming, re-generate plans for behaviors that have fewer than 10 total strategies. Implies --resume."
    )
    # ===== END: UPDATED ARGUMENT DEFINITION =====
    parsed_args = args.parse_args()

    # If --resume-empty is used, it implies --resume
    if parsed_args.resume_empty:
        parsed_args.resume = True

    config_path = parsed_args.config

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    apg_cfg = config["attack_plan_generator"]
    if parsed_args.language:
        apg_cfg["attack_language"] = parsed_args.language
    else:
        apg_cfg.setdefault("attack_language", "auto")

    base_out_dir = apg_cfg["attack_plan_generation_dir"]
    
    output_dir = None
    all_behaviors_data = []
    all_messages = []
    completed_behavior_ids = set()

    if parsed_args.resume:
        if os.path.exists(base_out_dir):
            all_subdirs = [d for d in os.listdir(base_out_dir) if os.path.isdir(os.path.join(base_out_dir, d))]
            timestamp_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$')
            valid_timestamp_dirs = sorted([d for d in all_subdirs if timestamp_pattern.match(d)], reverse=True)
            
            if valid_timestamp_dirs:
                latest_dir_name = valid_timestamp_dirs[0]
                output_dir = os.path.join(base_out_dir, latest_dir_name)

                attack_plans_path = os.path.join(output_dir, "attack_plans.json")
                messages_path = os.path.join(output_dir, "all_messages.json")

                if os.path.exists(attack_plans_path):
                    try:
                        with open(attack_plans_path, "r", encoding='utf-8') as f:
                            loaded_behaviors = json.load(f)
                        
                        if not isinstance(loaded_behaviors, list):
                            loaded_behaviors = []

                        ids_to_rerun = set()
                        if parsed_args.resume_empty:
                            temp_behaviors_data = []
                            for item in loaded_behaviors:
                                # ===== START: MODIFIED LOGIC FOR "LESS THAN 10" =====
                                # Count the total number of strategies for this behavior
                                total_strategies = 0
                                if 'attack_strategies' in item and isinstance(item['attack_strategies'], dict):
                                    for set_name, strategies_in_set in item['attack_strategies'].items():
                                        if isinstance(strategies_in_set, dict):
                                            total_strategies += len(strategies_in_set)
                                
                                # Check if the total is less than 10
                                if total_strategies < 10:
                                    ids_to_rerun.add(item['behavior_details']['BehaviorID'])
                                else:
                                    temp_behaviors_data.append(item)
                                # ===== END: MODIFIED LOGIC FOR "LESS THAN 10" =====
                            
                            if ids_to_rerun:
                                logging.warning(f"Found {len(ids_to_rerun)} behaviors with fewer than 10 strategies. They will be re-processed.")
                                all_behaviors_data = temp_behaviors_data
                            else:
                                all_behaviors_data = loaded_behaviors
                        else:
                            all_behaviors_data = loaded_behaviors

                        completed_behavior_ids = {
                            item['behavior_details']['BehaviorID']
                            for item in all_behaviors_data
                        }

                        if os.path.exists(messages_path):
                            with open(messages_path, "r", encoding='utf-8') as f:
                                loaded_messages = json.load(f)
                            if isinstance(loaded_messages, list):
                                if ids_to_rerun:
                                    all_messages = [msg for msg in loaded_messages if msg.get('behavior_id') not in ids_to_rerun]
                                else:
                                    all_messages = loaded_messages
                    except (json.JSONDecodeError, IOError) as e:
                        print(f"Warning: Could not read 'attack_plans.json': {e}. Starting fresh in directory.")
    
    if not output_dir:
        output_dir = create_output_directory(base_out_dir)
    
    setup_logging(output_dir)

    if parsed_args.resume and (completed_behavior_ids or all_behaviors_data):
        logging.info(f"Resuming run in directory: {output_dir}")
        logging.info(f"Loaded {len(completed_behavior_ids)} behaviors that meet the criteria.")
    else:
        logging.info(f"Starting new run in directory: {output_dir}")

    agent = BaseAgent(config["attack_plan_generator"])
    progress_dir = apg_cfg.get("progress_dir", os.path.join(base_out_dir, "progress"))
    os.makedirs(base_out_dir, exist_ok=True)
    os.makedirs(progress_dir, exist_ok=True)

    df_full = load_datasets(apg_cfg["behavior_path"], apg_cfg["num_behaviors"])

    if completed_behavior_ids:
        df = df_full[~df_full['BehaviorID'].isin(completed_behavior_ids)].copy()
    else:
        df = df_full

    if df.empty:
        logging.info("All specified behaviors have already been processed and meet the criteria. Nothing to do. Exiting.")
        return

    logging.info(f"Processing {len(df)} remaining behaviors.")

    all_params = [
        {
            "i": i,
            "row": row,
            "agent": agent,
            "temperature": apg_cfg["temperature"],
            "progress_dir": progress_dir,
            "default_grading_function": config.get("evaluation", {}).get("grading_function", "")
        }
        for i, row in df.iterrows()
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_single_behavior, **args): args for args in all_params}
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            try:
                behavior_result, messages_result = future.result()
                all_behaviors_data.append(behavior_result)
                all_messages.extend(messages_result)
                
                with open(os.path.join(output_dir, "attack_plans.json"), "w", encoding='utf-8') as f:
                    json.dump(all_behaviors_data, f, indent=4, ensure_ascii=False)

                with open(os.path.join(output_dir, "all_messages.json"), "w", encoding='utf-8') as f:
                    json.dump(all_messages, f, indent=4, ensure_ascii=False)
            except Exception as exc:
                logging.error(f'A behavior generation thread raised an exception: {exc}')

    logging.info(f"Finished. All results saved in {output_dir}")

if __name__ == "__main__":
    main()




# import argparse
# import concurrent.futures
# import json
# import logging
# import os
# from datetime import datetime
# import re
# from pprint import pformat
# import numpy as np
# import pandas as pd
# import tqdm
# import yaml
# from tenacity import retry, stop_after_attempt, wait_fixed
# from utils.helpers import extract_all_json_objects, normalize_strategies, merge_and_save

# from agents.base_agent import BaseAgent


# def setup_logging(output_dir):
#     """Setup logging to both file and console with ANSI code handling"""

#     class NoColorFormatter(logging.Formatter):
#         def format(self, record):
#             import re

#             record.msg = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", str(record.msg))
#             return super().format(record)

#     # Ensure the directory exists before setting up the file handler
#     os.makedirs(output_dir, exist_ok=True)
#     file_handler = logging.FileHandler(os.path.join(output_dir, "generation_log.txt"))
#     file_handler.setFormatter(
#         NoColorFormatter("%(asctime)s - %(threadName)s - %(levelname)s - %(message)s")
#     )

#     console_handler = logging.StreamHandler()
#     console_handler.setFormatter(logging.Formatter("%(message)s"))

#     # Remove existing handlers to avoid duplicate logs in case of re-setup
#     for handler in logging.root.handlers[:]:
#         logging.root.removeHandler(handler)
        
#     logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])

# def parse_functions_column(val):
#     """Return list[str] of exact tool names."""
#     if val is None:
#         return []
#     s = str(val).strip()
#     # Try JSON array first
#     try:
#         obj = json.loads(s)
#         if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
#             return obj
#     except Exception:
#         pass
#     # Fallback: split on commas
#     parts = [p.strip() for p in re.split(r'\s*,\s*', s) if p.strip()]
#     return parts
    
# def load_datasets(csv_path, number_of_behaviors=15):
#     """Load and filter dataset"""
#     np.random.seed(27)
#     df = pd.read_csv(csv_path)
#     filtered_df = df[df["FunctionalCategory"] == "standard"]
#     # If number_of_behaviors is larger than the df, just return the whole df
#     if number_of_behaviors > len(filtered_df):
#         return filtered_df
#     return filtered_df.sample(n=number_of_behaviors)

# def extract_json_from_output(raw_content: str) -> dict:
#     """
#     Extracts the first JSON object from a string, even if it's inside a code block.
#     Returns a Python dict. Raises ValueError if no JSON found.
#     """
#     # Find all {...} blocks (including nested ones)
#     matches = re.findall(r'```json\s*([\s\S]*?)```', raw_content)
#     if matches:
#         json_str = matches[0]
#     else:
#         # Try to find first { ... } block
#         matches = re.findall(r'({[\s\S]+})', raw_content)
#         if matches:
#             json_str = matches[0]
#         else:
#             raise ValueError("No JSON object found in model output.")
#     # Try to parse
#     try:
#         return json.loads(json_str)
#     except Exception as e:
#         raise ValueError(f"JSON parsing failed: {e}\nContent: {json_str}")

# def load_and_format_prompts(behavior, set_number, previous_responses=None,
#                             attack_language="auto", target_functions=None):
#     with open("config/prompts/plan_generation_prompts.yaml", "r") as f:
#         prompts = yaml.safe_load(f)

#     system_prompt = prompts["prompts"]["system"]["messages"][0]["content"]
#     system_prompt = system_prompt.replace("{attack_language}", attack_language)

#     # Prepare tool placeholders
#     target_functions = target_functions or []
#     tools_csv = ", ".join(target_functions)
#     tools_json = json.dumps(target_functions, ensure_ascii=False)

#     if set_number == 1:
#         user_prompt = prompts["prompts"]["user_message1"]["messages"][0]["content"]
#         formatted_user_prompt = (
#             user_prompt
#             .replace("{target_behavior}", behavior)
#             .replace("{attack_language}", attack_language)
#             .replace("{target_functions_comma_separated}", tools_csv)
#             .replace("{target_tools_json}", tools_json)
#         )
#     else:
#         user_prompt = prompts["prompts"]["user_message2"]["messages"][0]["content"]
#         formatted_user_prompt = (
#             user_prompt
#             .replace("{target_behavior}", behavior)
#             .replace("{attack_language}", attack_language)
#             .replace("{target_functions_comma_separated}", tools_csv)
#             .replace("{target_tools_json}", tools_json)
#         )

#         strategies_text = ""
#         for set_name, response in (previous_responses or {}).items():
#             strategies_text += f"\n{set_name}:\n{json.dumps(response, ensure_ascii=False)}\n"
#         formatted_user_prompt = formatted_user_prompt.replace(
#             "{previously_generated_strategies}", strategies_text or "{}"
#         )

#     return system_prompt, formatted_user_prompt



# def create_output_directory(base_output_dir):
#     """Create timestamped output directory"""
#     timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     output_dir = os.path.join(base_output_dir, timestamp)
#     os.makedirs(output_dir, exist_ok=True)
#     return output_dir


# @retry(stop=stop_after_attempt(5), wait=wait_fixed(1))
# def generate_strategies(agent, messages, set_num, temperature, *, behavior_id, progress_dir):
#     """Call model, salvage any valid strategies, persist immediately, return what we kept."""
#     raw = agent.call_api(
#         messages=messages,
#         temperature=temperature,
#         response_format={"type": "json_object"},
#     )

#     PROGRESS_PATH = os.path.join(progress_dir, "attack_plans_progress.json")
#     good_total = 0
#     combined = {}

#     # --- Robust handling of different possible return types ---
#     if raw is None:
#         # Raise to trigger tenacity retry
#         raise ValueError("Model returned None for this request")

#     if isinstance(raw, dict):
#         # Some providers (or JSON mode) return a parsed dict payload
#         obj_iter = [raw]
#     else:
#         # Ensure string
#         if not isinstance(raw, str):
#             # try to coerce to string if it's something else
#             raw = json.dumps(raw, ensure_ascii=False)
#         if not raw.strip():
#             raise ValueError("Empty string returned from model")
#         # extract_all_json_objects() expects a string and returns a list of dicts
#         obj_iter = extract_all_json_objects(raw)

#     # --- Process each extracted JSON object ---
#     for obj in obj_iter:
#         picked = normalize_strategies(obj)  # keep only strategy_* and schema-valid
#         if not picked:
#             continue
#         added = merge_and_save(PROGRESS_PATH, behavior_id, picked)  # atomic merge
#         good_total += added
#         combined.update(picked)

#     if good_total == 0:
#         logging.warning(f"[warn] no valid strategies salvaged for behavior {behavior_id} in set {set_num}; response may have been a refusal or malformed.")
#     else:
#         logging.info(f"[info] saved {good_total} strategy(ies) for behavior {behavior_id}.")

#     logging.info(f"\nSet {set_num} Generated Strategies (kept {len(combined)}):")
#     logging.info(json.dumps(combined, indent=2, ensure_ascii=False))

#     return combined

    
# def process_single_behavior(i, row, agent, temperature, progress_dir, num_sets=5):
#     behavior = row["Behavior"]
#     behavior_id = row["BehaviorID"]
#     logging.info(f"\n{'='*50}")
#     logging.info(f"Processing Behavior {i} (ID: {behavior_id}):")
#     logging.info(f"Behavior: {behavior}")
#     logging.info(f"{'='*50}")

#     all_messages = []

#     # Initialize behavior data
#     all_responses = {}
#     behavior_details = {
#         k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()
#     }

#     csv_gf = (
#     behavior_details.get("grading_function")
#     or behavior_details.get("GradingFunction")
#     or behavior_details.get("gradingFunction")
# )
#     behavior_data = {
#         "behavior_number": i,
#         "behavior_details": behavior_details,
#         "attack_strategies": all_responses,
#     }
#     functions_raw = row.get("Functions") if "Functions" in row else None
#     target_functions = parse_functions_column(functions_raw)

#     behavior_details["grading_function"] = csv_gf

#     # Generate strategies for each set
#     for set_num in range(1, num_sets + 1):
#         logging.info(f"\nGenerating Set {set_num}:")

#         system_prompt, formatted_user_prompt = load_and_format_prompts(
#         behavior=behavior,
#         set_number=set_num,
#         previous_responses=all_responses,
#         target_functions=target_functions,
#         attack_language=agent.config.get("attack_language", "auto")  # from config
#     )

#         messages = [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": formatted_user_prompt},
#         ]

#         logging.info(f"Messages for set {set_num}")
#         logging.info(pformat(messages, indent=2))  # instead of pprint(...)


#         message_data = {
#             "behavior_index": i,
#             "behavior_id": behavior_id, # Add behavior_id for easier filtering
#             "behavior": behavior,
#             "set_number": set_num,
#             "messages": messages,
#         }
#         all_messages.append(message_data)

#         response = generate_strategies(
#             agent=agent,
#             messages=messages,
#             set_num=set_num,
#             temperature=temperature,
#             behavior_id=behavior_id,
#             progress_dir=progress_dir,
#         )

#         all_responses[f"Set_{set_num}"] = response

#     return behavior_data, all_messages


# def main():
#     args = argparse.ArgumentParser(
#         description="Generates multi-turn jailbreak attack strategies for X-Teaming."
#     )
#     args.add_argument(
#         "-c", "--config", action="store", type=str, default="./config/config.yaml"
#     )
#     args.add_argument(
#         "-L", "--language", default=None,
#         help="Language for conversation snippets in the plan (e.g. 'Hindi', 'French', 'English', or 'auto')."
#     )
#     args.add_argument(
#         "-r", "--resume", action="store_true",
#         help="Resume the latest unfinished attack plan generation run."
#     )
#     # ===== START: UPDATED ARGUMENT DEFINITION =====
#     args.add_argument(
#         "--resume-empty", action="store_true",
#         help="When resuming, re-generate plans for behaviors that have fewer than 10 total strategies. Implies --resume."
#     )
#     # ===== END: UPDATED ARGUMENT DEFINITION =====
#     parsed_args = args.parse_args()

#     # If --resume-empty is used, it implies --resume
#     if parsed_args.resume_empty:
#         parsed_args.resume = True

#     config_path = parsed_args.config

#     with open(config_path, "r") as f:
#         config = yaml.safe_load(f)

#     apg_cfg = config["attack_plan_generator"]
#     if parsed_args.language:
#         apg_cfg["attack_language"] = parsed_args.language
#     else:
#         apg_cfg.setdefault("attack_language", "auto")

#     base_out_dir = apg_cfg["attack_plan_generation_dir"]
    
#     output_dir = None
#     all_behaviors_data = []
#     all_messages = []
#     completed_behavior_ids = set()

#     if parsed_args.resume:
#         if os.path.exists(base_out_dir):
#             all_subdirs = [d for d in os.listdir(base_out_dir) if os.path.isdir(os.path.join(base_out_dir, d))]
#             timestamp_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$')
#             valid_timestamp_dirs = sorted([d for d in all_subdirs if timestamp_pattern.match(d)], reverse=True)
            
#             if valid_timestamp_dirs:
#                 latest_dir_name = valid_timestamp_dirs[0]
#                 output_dir = os.path.join(base_out_dir, latest_dir_name)

#                 attack_plans_path = os.path.join(output_dir, "attack_plans.json")
#                 messages_path = os.path.join(output_dir, "all_messages.json")

#                 if os.path.exists(attack_plans_path):
#                     try:
#                         with open(attack_plans_path, "r", encoding='utf-8') as f:
#                             loaded_behaviors = json.load(f)
                        
#                         if not isinstance(loaded_behaviors, list):
#                             loaded_behaviors = []

#                         ids_to_rerun = set()
#                         if parsed_args.resume_empty:
#                             temp_behaviors_data = []
#                             for item in loaded_behaviors:
#                                 # ===== START: MODIFIED LOGIC FOR "LESS THAN 10" =====
#                                 # Count the total number of strategies for this behavior
#                                 total_strategies = 0
#                                 if 'attack_strategies' in item and isinstance(item['attack_strategies'], dict):
#                                     for set_name, strategies_in_set in item['attack_strategies'].items():
#                                         if isinstance(strategies_in_set, dict):
#                                             total_strategies += len(strategies_in_set)
                                
#                                 # Check if the total is less than 10
#                                 if total_strategies < 10:
#                                     ids_to_rerun.add(item['behavior_details']['BehaviorID'])
#                                 else:
#                                     temp_behaviors_data.append(item)
#                                 # ===== END: MODIFIED LOGIC FOR "LESS THAN 10" =====
                            
#                             if ids_to_rerun:
#                                 logging.warning(f"Found {len(ids_to_rerun)} behaviors with fewer than 10 strategies. They will be re-processed.")
#                                 all_behaviors_data = temp_behaviors_data
#                             else:
#                                 all_behaviors_data = loaded_behaviors
#                         else:
#                             all_behaviors_data = loaded_behaviors

#                         completed_behavior_ids = {
#                             item['behavior_details']['BehaviorID']
#                             for item in all_behaviors_data
#                         }

#                         if os.path.exists(messages_path):
#                             with open(messages_path, "r", encoding='utf-8') as f:
#                                 loaded_messages = json.load(f)
#                             if isinstance(loaded_messages, list):
#                                 if ids_to_rerun:
#                                     all_messages = [msg for msg in loaded_messages if msg.get('behavior_id') not in ids_to_rerun]
#                                 else:
#                                     all_messages = loaded_messages
#                     except (json.JSONDecodeError, IOError) as e:
#                         print(f"Warning: Could not read 'attack_plans.json': {e}. Starting fresh in directory.")
    
#     if not output_dir:
#         output_dir = create_output_directory(base_out_dir)
    
#     setup_logging(output_dir)

#     if parsed_args.resume and (completed_behavior_ids or all_behaviors_data):
#         logging.info(f"Resuming run in directory: {output_dir}")
#         logging.info(f"Loaded {len(completed_behavior_ids)} behaviors that meet the criteria.")
#     else:
#         logging.info(f"Starting new run in directory: {output_dir}")

#     agent = BaseAgent(config["attack_plan_generator"])
#     progress_dir = apg_cfg.get("progress_dir", os.path.join(base_out_dir, "progress"))
#     os.makedirs(base_out_dir, exist_ok=True)
#     os.makedirs(progress_dir, exist_ok=True)

#     df_full = load_datasets(apg_cfg["behavior_path"], apg_cfg["num_behaviors"])

#     if completed_behavior_ids:
#         df = df_full[~df_full['BehaviorID'].isin(completed_behavior_ids)].copy()
#     else:
#         df = df_full

#     if df.empty:
#         logging.info("All specified behaviors have already been processed and meet the criteria. Nothing to do. Exiting.")
#         return

#     logging.info(f"Processing {len(df)} remaining behaviors.")

#     all_params = [
#         {"i": i, "row": row, "agent": agent, "temperature": apg_cfg["temperature"], "progress_dir": progress_dir, "default_grading_function": config.get("evaluation", {}).get("grading_function", "")}
#         for i, row in df.iterrows()
#     ]

#     with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
#         futures = {executor.submit(process_single_behavior, **args): args for args in all_params}
#         for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
#             try:
#                 behavior_result, messages_result = future.result()
#                 all_behaviors_data.append(behavior_result)
#                 all_messages.extend(messages_result)
                
#                 with open(os.path.join(output_dir, "attack_plans.json"), "w", encoding='utf-8') as f:
#                     json.dump(all_behaviors_data, f, indent=4, ensure_ascii=False)

#                 with open(os.path.join(output_dir, "all_messages.json"), "w", encoding='utf-8') as f:
#                     json.dump(all_messages, f, indent=4, ensure_ascii=False)
#             except Exception as exc:
#                 logging.error(f'A behavior generation thread raised an exception: {exc}')

#     logging.info(f"Finished. All results saved in {output_dir}")

# if __name__ == "__main__":
#     main()