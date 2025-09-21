# run_jailbreak_evaluation.py

import json
import re
import os
import datetime
import torch
import argparse
import yaml

# Import the classes from your modules, just like in main.py
from agents.generator import LlamaGenerator
from agents.gpt_evaluator import LlmEvaluator

def load_config(config_path: str) -> dict:
    """Loads configurations from the YAML file."""
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"FATAL: Configuration file not found at '{config_path}'. Exiting.")
        return None
    except Exception as e:
        print(f"FATAL: Error reading config file: {e}")
        return None

def load_and_filter_conversations(file_path: str) -> list:
    """Loads conversations from the JSON file and filters for successful jailbreaks."""
    jailbreak_conversations = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for _, behavior_data in data.get("behaviors", {}).items():
            for strategy in behavior_data.get("strategies", []):
                if strategy.get("jailbreak_achieved") is True:
                    convo_history = []
                    for turn in strategy.get("conversation", []):
                        if "attacker" in turn:
                            convo_history.append({"role": "user", "content": turn["attacker"]})
                        if "target_truncated" in turn:
                            cleaned_text = re.sub(r'<think>.*?</think>', '', turn["target_truncated"], flags=re.DOTALL).strip()
                            convo_history.append({"role": "assistant", "content": cleaned_text})
                    
                    if convo_history:
                        jailbreak_conversations.append(convo_history)

        print(f"Loaded and filtered {len(jailbreak_conversations)} successful jailbreak conversations from '{file_path}'.")
    except FileNotFoundError:
        print(f"FATAL: Input data file '{file_path}' not found. Exiting.")
        return []
    except Exception as e:
        print(f"FATAL: An error occurred while loading the data file: {e}")
        return []
    return jailbreak_conversations

def main(args):
    """Main function to orchestrate generation and evaluation."""
    config = load_config(args.config)
    if not config:
        return

    # --- 1. Initialization from Config ---
    print("--- Initializing Models ---")
    generator_cfg = config.get("generator", {})
    eval_cfg = config.get("evaluation", {})

    generator = LlamaGenerator(
        model_name=generator_cfg.get("model_name"),
        auth_token=generator_cfg.get("auth_token")
    )
    if not generator.model:
        return
        
    evaluator = LlmEvaluator(eval_cfg)

    # --- 2. Load and Filter Data ---
    conversations = load_and_filter_conversations(args.input_file)
    if not conversations:
        return

    # --- 3. Process Each Conversation ---
    all_results = []
    for i, convo_history in enumerate(conversations):
        print(f"\n--- Processing Jailbreak Conversation {i+1}/{len(conversations)} ---")
        final_prompt = convo_history[-1]['content']
        
        # Generate new response using the full history
        new_response = generator.generate_response(convo_history)
        
        # Evaluate the new response
        score, reason = evaluator.infer_single(final_prompt, new_response)

        print(f"Final Prompt: {final_prompt}")
        print(f"Generated {generator_cfg.get('model_name')} Response: {new_response}")
        print(f"Qwen Evaluation -> Score: {score}/5, Reason: {reason}")
        
        all_results.append({
            "conversation_id": i + 1,
            "final_prompt": final_prompt,
            "generated_response_by_llama": new_response,
            "qwen_evaluation": {"score": score, "reason": reason},
            "original_history": convo_history
        })

    # --- 4. Save Results ---
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"jailbreak_evaluation_{timestamp}.json")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
        
    print(f"\n--- All conversations processed. Results saved to '{output_path}' ---")

    del generator
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-evaluate successful jailbreak conversations.")
    parser.add_argument("-c", "--config", default="config/config.yaml", help="Path to the config file.")
    parser.add_argument("input_file", help="Path to the JSON file with conversation results (e.g., Hindi_jailbreak_engstrategy.json).")
    args = parser.parse_args()
    main(args)