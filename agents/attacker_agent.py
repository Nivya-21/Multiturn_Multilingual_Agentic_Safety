# agents/attacker_agent.py
import json
import yaml
from typing import Dict, Optional
from tenacity import retry, stop_after_attempt, wait_fixed

from agents.base_agent import BaseAgent
from agents.target_model import TargetModel, TGTargetModel, truncate_response
from utils.sanitize import strip_reasoning

class LLMRefusalError(Exception):
    pass

class AttackerAgent(BaseAgent):
    """
    Attacker:
    - treats each turn as a phase
    - can receive a safe phase-level feedback block from main
    - injects that into the user prompt as [PHASE_FEEDBACK] ... [/PHASE_FEEDBACK]
    """
    def __init__(self, config: Dict, behavior: Dict, strategy: Dict):
        super().__init__(config)
        self.messages = []
        self._initialize_agent(behavior, strategy)
        self._verifier_feedback: Optional[Dict] = None

    # def set_verifier_feedback(self, details: Optional[Dict]) -> None:
    #     self._verifier_feedback = details or {}

    # def _format_verifier_hint(self) -> str:
    #     if not self._verifier_feedback:
    #         return ""
    #     return json.dumps(self._verifier_feedback, ensure_ascii=False)


    def set_verifier_feedback(self, details: Optional[Dict]) -> None:
        # main passes phase-scoped {phase_status, phase_progress, hint}
        self._verifier_feedback = details or {}

    def _format_verifier_hint(self) -> str:
        if not self._verifier_feedback:
            return ""
        try:
            return json.dumps(self._verifier_feedback, ensure_ascii=False)
        except Exception:
            return str(self._verifier_feedback)


    def _initialize_agent(self, behavior: Dict, strategy: Dict) -> None:
        self.behavior = behavior["Behavior"]
        self.strategy_dict = strategy
        self.strategy = self._format_strategy(strategy)

        plan = strategy["conversation_plan"]
        turn_keys = [k for k in plan.keys() if k.startswith("turn_")]
        has_final = "final_turn" in plan
        self._has_final_turn = has_final
        self.num_phases = len(turn_keys) + (1 if has_final else 0)

        self.plan_phase = 1
        self.plan_score_history = []
        self.next_message_candidates = []
        self._load_prompt_templates()

    def _format_strategy(self, strategy: Dict) -> str:
        return "\n".join(
            [
                f"Persona: {strategy['persona']}",
                f"Context: {strategy['context']}",
                f"Approach: {strategy['approach']}",
                f"Conversation Plan: {strategy['conversation_plan']}",
            ]
        )

    def _load_prompt_templates(self) -> dict:
        with open("config/prompts/attacker_agent_prompts.yaml", "r") as f:
            prompts = yaml.safe_load(f)["prompts"]
        self.system_prompt = prompts["system"]["messages"][0]["content"]
        self.first_turn_template = prompts["first_turn_messages"]["messages"][0]["content"]
        self.nth_turn_template = prompts["nth_turn_messages"]["messages"][0]["content"]
        self.final_turn_template = prompts["final_turn_messages"]["messages"][0]["content"]
        self.revise_plan_template = prompts["revise_plan_messages"]["messages"][0]["content"]
        return prompts

    def commit(self):
        """
        Always advance a phase, including after the final phase.
        This allows main.py to detect plan_phase > num_phases and stop cleanly.
        """
        if not self.next_message_candidates:
            raise ValueError("A response has not been generated yet!")

        best = max(self.next_message_candidates, key=lambda x: x["score"])
        self.messages += best["messages"]
        self.plan_score_history.append(best["score"])
        self.plan_phase += 1
        self.next_message_candidates.clear()

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1))
    def generate_response(self, conversation_history: str = None) -> str:
        if self.plan_phase == 1:
            response = self._generate_first_turn()
        elif self.plan_phase == self.num_phases and self._has_final_turn:
            response = self._generate_final_turn(conversation_history)
        else:
            response = self._generate_nth_turn(conversation_history)
        return response

    def generate_and_evaluate_candidate(
        self,
        target: TargetModel,
        conversation_history: str = None,
    ) -> dict:
        attacker_msg = self.generate_response(conversation_history)
        target_response = target.generate_response(attacker_msg) or ""
        clean = strip_reasoning(target_response)
        truncated = truncate_response(clean)

        next_messages = [
            {
                "role": "system",
                "content": self.system_prompt.format(
                    target_behavior=self.behavior,
                    strategy=self.strategy,
                    max_turns=self.num_phases,
                    attack_language=self.config.get("attack_language", "auto"),
                ),
            },
            {"role": "user", "content": attacker_msg},
            {"role": "assistant", "content": clean},
        ]
        self.next_message_candidates.append({"score": 0, "messages": next_messages})

        payload = {
            "phase": self.plan_phase,
            "attacker": attacker_msg,
            "target": clean,
            "target_truncated": truncated,
            "evaluation_score": 0,
            "evaluation_reason": "Pending AgentHarm verification.",
        }
        try:
            if hasattr(target, "get_inspect_messages"):
                payload["inspect_messages"] = target.get_inspect_messages()
        except Exception:
            pass
        return payload

    def _generate_first_turn(self) -> str:
        turn_1_conversation_flow = self.strategy_dict["conversation_plan"]["turn_1"]

        formatted_system_prompt = self.system_prompt.format(
            target_behavior=self.behavior,
            strategy=self.strategy,
            max_turns=self.num_phases,
            attack_language=self.config.get("attack_language", "auto"),
        )
        formatted_user_prompt = self.first_turn_template.format(
            turn_1_conversation_flow=turn_1_conversation_flow,
            attack_language=self.config.get("attack_language", "auto"),
        )

        vf = self._format_verifier_hint()
        if vf:
            formatted_user_prompt += f"\n\n[PHASE_FEEDBACK]\n{vf}\n[/PHASE_FEEDBACK]"

        next_messages = [
            {"role": "system", "content": formatted_system_prompt},
            {"role": "user", "content": formatted_user_prompt},
        ]

        response = self.call_api(
            messages=next_messages, temperature=self.config["temperature"]
        )
        next_messages.append({"role": "assistant", "content": response})
        self.next_message_candidates.append({"score": 0, "messages": next_messages})

        return self._extract_conversation(response)

    def _generate_nth_turn(self, conversation_history: str) -> str:
        if self.plan_phase >= self.num_phases:
            raise ValueError(
                f"Expected plan phase < {self.num_phases}, but we are on phase {self.plan_phase}"
            )

        turn_key = f"turn_{self.plan_phase}"
        turn_conversation_flow = self.strategy_dict["conversation_plan"][turn_key]

        formatted_prompt = self.nth_turn_template.format(
            turn_number=self.plan_phase,
            conversation_history=conversation_history or "",
            turn_number_conversation_flow=turn_conversation_flow,
            previous_turn_number=self.plan_phase - 1,
            attack_language=self.config.get("attack_language", "auto"),
        )

        vf = self._format_verifier_hint()
        if vf:
            formatted_prompt += f"\n\n[PHASE_FEEDBACK]\n{vf}\n[/PHASE_FEEDBACK]"

        next_messages = [{"role": "user", "content": formatted_prompt}]

        formatted_system_prompt = self.system_prompt.format(
            target_behavior=self.behavior,
            strategy=self.strategy,
            max_turns=self.num_phases,
            attack_language=self.config.get("attack_language", "auto"),
        )
        sys_prompt = [{"role": "system", "content": formatted_system_prompt}]

        response = self.call_api(
            sys_prompt + next_messages, temperature=self.config["temperature"]
        )
        next_messages.append({"role": "assistant", "content": response})
        self.next_message_candidates.append({"score": 0, "messages": next_messages})

        return self._extract_conversation(response)

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1))
    def revise_plan(self, conversation_history: str = ""):
        formatted_system_prompt = self.system_prompt.format(
            target_behavior=self.behavior,
            strategy=self.strategy,
            max_turns="as many turns as necessary",
            attack_language=self.config.get("attack_language", "auto"),
        )

        formatted_prompt = (
            self.revise_plan_template.replace("{target_behavior}", self.behavior)
            .replace("{turn_number}", str(self.plan_phase))
            .replace("{conversation_history}", conversation_history or "")
            .replace("{old_plan}", json.dumps(self.strategy_dict, indent=4))
            .replace("{previous_turn_number}", str(self.plan_phase - 1))
        )

        response = self.call_api(
            [
                {"role": "system", "content": formatted_system_prompt},
                {"role": "user", "content": formatted_prompt},
            ],
            temperature=self.config["temperature"],
        )

        response_slice = response[response.find("{"): response.rfind("}") + 1]
        new_strategy = json.loads(response_slice)

        plan = new_strategy.get("conversation_plan", {})
        if not isinstance(plan, dict):
            raise ValueError("Revised plan's 'conversation_plan' is missing or not a dictionary.")

        turn_keys = [k for k in plan.keys() if k.startswith("turn_")]
        final_turn_exists = "final_turn" in plan
        if not turn_keys and not final_turn_exists:
            raise ValueError("Revised plan has no turns ('turn_X' or 'final_turn').")

        self.num_phases = len(turn_keys) + (1 if final_turn_exists else 0)
        self._has_final_turn = final_turn_exists

        self.strategy_dict = new_strategy
        self.strategy = self._format_strategy(new_strategy)
        return self.strategy_dict

    def _generate_final_turn(self, conversation_history: str) -> str:
        plan = self.strategy_dict.get("conversation_plan", {})
        final_turn_conversation_flow = plan.get("final_turn")
        if not final_turn_conversation_flow:
            return ""

        formatted_prompt = self.final_turn_template.format(
            conversation_history=conversation_history or "",
            final_turn_conversation_flow=final_turn_conversation_flow,
            attack_language=self.config.get("attack_language", "auto"),
        )

        vf = self._format_verifier_hint()
        if vf:
            # formatted_prompt += f"\n\n[PHASE_FEEDBACK]\n{vf}\n[/PHASE_FEEDBACK]"
            formatted_prompt += f"\n\n[PHASE_FEEDBACK]\n{vf}\n[/PHASE_FEEDBACK]"
            # NEW: make the last message terse when verifier says complete
            try:
                fb = json.loads(vf)
                if fb.get("phase_status") == "complete":
                    formatted_prompt += (
                        "\n\n[FINAL_STYLE]\n"
                        "Output ONLY one short confirmation sentence. "
                        "No offers of help, no new requests, no pleasantries.\n"
                        "[/FINAL_STYLE]"
                    )
            except Exception:
                pass

        formatted_system_prompt = self.system_prompt.format(
            target_behavior=self.behavior,
            strategy=self.strategy,
            max_turns=self.num_phases,
            attack_language=self.config.get("attack_language", "auto"),
        )
        sys_prompt = [{"role": "system", "content": formatted_system_prompt}]
        next_messages = [{"role": "user", "content": formatted_prompt}]

        response = self.call_api(
            sys_prompt + next_messages, temperature=self.config["temperature"]
        )
        next_messages.append({"role": "assistant", "content": response})
        self.next_message_candidates.append({"score": 0, "messages": next_messages})

        return self._extract_conversation(response)

    def _extract_conversation(self, response: str) -> str:
        start = response.find("<conversation>") + len("<conversation>")
        end = response.find("</conversation>")
        return response[start:end].strip()

class TGAttackerAgent(AttackerAgent):
    def __init__(self, attacker_config: Dict, target_config: Dict, eval_config: Dict, behavior: Dict, strategy: Dict):
        super().__init__(attacker_config, behavior, strategy)
        self.last_phase = 0
        self.target_agent = TGTargetModel(target_config)

# # agents/attacker_agent.py
# import json
# import yaml
# from typing import Dict, Optional
# from tenacity import retry, stop_after_attempt, wait_fixed

# from agents.base_agent import BaseAgent
# from agents.target_model import TargetModel, TGTargetModel, truncate_response
# from utils.sanitize import strip_reasoning


# class LLMRefusalError(Exception):
#     pass


# class AttackerAgent(BaseAgent):
#     """
#     Attacker now:
#     - treats each turn as a phase
#     - can receive a safe phase-level feedback block from main
#     - injects that into the user prompt as [PHASE_FEEDBACK] ... [/PHASE_FEEDBACK]
#     """
#     def __init__(self, config: Dict, behavior: Dict, strategy: Dict):
#         super().__init__(config)
#         self.messages = []
#         self._initialize_agent(behavior, strategy)
#         self._verifier_feedback: Optional[Dict] = None

#     def set_verifier_feedback(self, details: Optional[Dict]) -> None:
#         # main will pass only phase, status, progress, hint
#         self._verifier_feedback = details or {}

#     def _format_verifier_hint(self) -> str:
#         if not self._verifier_feedback:
#             return ""
#         return json.dumps(self._verifier_feedback, ensure_ascii=False)

#     def _initialize_agent(self, behavior: Dict, strategy: Dict) -> None:
#         self.behavior = behavior["Behavior"]
#         self.strategy_dict = strategy
#         self.strategy = self._format_strategy(strategy)

#         plan = strategy["conversation_plan"]
#         turn_keys = [k for k in plan.keys() if k.startswith("turn_")]
#         has_final = "final_turn" in plan
#         self._has_final_turn = has_final
#         self.num_phases = len(turn_keys) + (1 if has_final else 0)

#         self.plan_phase = 1
#         self.plan_score_history = []
#         self.next_message_candidates = []
#         self._load_prompt_templates()

#     def _format_strategy(self, strategy: Dict) -> str:
#         return "\n".join(
#             [
#                 f"Persona: {strategy['persona']}",
#                 f"Context: {strategy['context']}",
#                 f"Approach: {strategy['approach']}",
#                 f"Conversation Plan: {strategy['conversation_plan']}",
#             ]
#         )

#     def _load_prompt_templates(self) -> dict:
#         with open("config/prompts/attacker_agent_prompts.yaml", "r") as f:
#             prompts = yaml.safe_load(f)["prompts"]
#         self.system_prompt = prompts["system"]["messages"][0]["content"]
#         self.first_turn_template = prompts["first_turn_messages"]["messages"][0]["content"]
#         self.nth_turn_template = prompts["nth_turn_messages"]["messages"][0]["content"]
#         self.final_turn_template = prompts["final_turn_messages"]["messages"][0]["content"]
#         self.revise_plan_template = prompts["revise_plan_messages"]["messages"][0]["content"]
#         return prompts

#     def commit(self):
#         """
#         Always advance a phase, including after the final phase.
#         This allows main.py to detect plan_phase > num_phases and stop cleanly.
#         """
#         if not self.next_message_candidates:
#             raise ValueError("A response has not been generated yet!")

#         best = max(self.next_message_candidates, key=lambda x: x["score"])
#         self.messages += best["messages"]
#         self.plan_score_history.append(best["score"])
#         self.plan_phase += 1
#         self.next_message_candidates.clear()

#     @retry(stop=stop_after_attempt(5), wait=wait_fixed(1))
#     def generate_response(self, conversation_history: str = None) -> str:
#         if self.plan_phase == 1:
#             response = self._generate_first_turn()
#         elif self.plan_phase == self.num_phases and self._has_final_turn:
#             response = self._generate_final_turn(conversation_history)
#         else:
#             response = self._generate_nth_turn(conversation_history)
#         return response

#     def generate_and_evaluate_candidate(
#         self,
#         target: TargetModel,
#         conversation_history: str = None,
#     ) -> dict:
#         attacker_msg = self.generate_response(conversation_history)
#         target_response = target.generate_response(attacker_msg) or ""
#         clean = strip_reasoning(target_response)
#         truncated = truncate_response(clean)

#         next_messages = [
#             {
#                 "role": "system",
#                 "content": self.system_prompt.format(
#                     target_behavior=self.behavior,
#                     strategy=self.strategy,
#                     max_turns=self.num_phases,
#                     attack_language=self.config.get("attack_language", "auto"),
#                 ),
#             },
#             {"role": "user", "content": attacker_msg},
#             {"role": "assistant", "content": clean},
#         ]
#         self.next_message_candidates.append({"score": 0, "messages": next_messages})

#         payload = {
#             "phase": self.plan_phase,
#             "attacker": attacker_msg,
#             "target": clean,
#             "target_truncated": truncated,
#             "evaluation_score": 0,
#             "evaluation_reason": "Pending AgentHarm verification.",
#         }
#         try:
#             if hasattr(target, "get_inspect_messages"):
#                 payload["inspect_messages"] = target.get_inspect_messages()
#         except Exception:
#             pass
#         return payload

#     def _generate_first_turn(self) -> str:
#         turn_1_conversation_flow = self.strategy_dict["conversation_plan"]["turn_1"]

#         formatted_system_prompt = self.system_prompt.format(
#             target_behavior=self.behavior,
#             strategy=self.strategy,
#             max_turns=self.num_phases,
#             attack_language=self.config.get("attack_language", "auto"),
#         )
#         formatted_user_prompt = self.first_turn_template.format(
#             turn_1_conversation_flow=turn_1_conversation_flow,
#             attack_language=self.config.get("attack_language", "auto"),
#         )

#         vf = self._format_verifier_hint()
#         if vf:
#             formatted_user_prompt += f"\n\n[PHASE_FEEDBACK]\n{vf}\n[/PHASE_FEEDBACK]"

#         next_messages = [
#             {"role": "system", "content": formatted_system_prompt},
#             {"role": "user", "content": formatted_user_prompt},
#         ]

#         response = self.call_api(
#             messages=next_messages, temperature=self.config["temperature"]
#         )
#         next_messages.append({"role": "assistant", "content": response})
#         self.next_message_candidates.append({"score": 0, "messages": next_messages})

#         return self._extract_conversation(response)

#     def _generate_nth_turn(self, conversation_history: str) -> str:
#         if self.plan_phase >= self.num_phases:
#             raise ValueError(
#                 f"Expected plan phase < {self.num_phases}, but we are on phase {self.plan_phase}"
#             )

#         turn_key = f"turn_{self.plan_phase}"
#         turn_conversation_flow = self.strategy_dict["conversation_plan"][turn_key]

#         formatted_prompt = self.nth_turn_template.format(
#             turn_number=self.plan_phase,
#             conversation_history=conversation_history,
#             turn_number_conversation_flow=turn_conversation_flow,
#             previous_turn_number=self.plan_phase - 1,
#             attack_language=self.config.get("attack_language", "auto"),
#         )

#         vf = self._format_verifier_hint()
#         if vf:
#             formatted_prompt += f"\n\n[PHASE_FEEDBACK]\n{vf}\n[/PHASE_FEEDBACK]"

#         next_messages = [{"role": "user", "content": formatted_prompt}]

#         formatted_system_prompt = self.system_prompt.format(
#             target_behavior=self.behavior,
#             strategy=self.strategy,
#             max_turns=self.num_phases,
#             attack_language=self.config.get("attack_language", "auto"),
#         )
#         sys_prompt = [{"role": "system", "content": formatted_system_prompt}]

#         response = self.call_api(
#             sys_prompt + next_messages, temperature=self.config["temperature"]
#         )
#         next_messages.append({"role": "assistant", "content": response})
#         self.next_message_candidates.append({"score": 0, "messages": next_messages})

#         return self._extract_conversation(response)

#     @retry(stop=stop_after_attempt(5), wait=wait_fixed(1))
#     def revise_plan(self, conversation_history: str = ""):
#         # kept for compatibility
#         formatted_system_prompt = self.system_prompt.format(
#             target_behavior=self.behavior,
#             strategy=self.strategy,
#             max_turns="as many turns as necessary",
#             attack_language=self.config.get("attack_language", "auto"),
#         )

#         formatted_prompt = (
#             self.revise_plan_template.replace("{target_behavior}", self.behavior)
#             .replace("{turn_number}", str(self.plan_phase))
#             .replace("{conversation_history}", conversation_history or "")
#             .replace("{old_plan}", json.dumps(self.strategy_dict, indent=4))
#             .replace("{previous_turn_number}", str(self.plan_phase - 1))
#         )

#         response = self.call_api(
#             [
#                 {"role": "system", "content": formatted_system_prompt},
#                 {"role": "user", "content": formatted_prompt},
#             ],
#             temperature=self.config["temperature"],
#         )

#         # try to parse a new strategy, but if it fails, raise
#         response_slice = response[response.find("{"): response.rfind("}") + 1]
#         new_strategy = json.loads(response_slice)

#         plan = new_strategy.get("conversation_plan", {})
#         if not isinstance(plan, dict):
#             raise ValueError("Revised plan's 'conversation_plan' is missing or not a dictionary.")

#         turn_keys = [k for k in plan.keys() if k.startswith("turn_")]
#         final_turn_exists = "final_turn" in plan
#         if not turn_keys and not final_turn_exists:
#             raise ValueError("Revised plan has no turns ('turn_X' or 'final_turn').")

#         self.num_phases = len(turn_keys) + (1 if final_turn_exists else 0)
#         self._has_final_turn = final_turn_exists

#         self.strategy_dict = new_strategy
#         self.strategy = self._format_strategy(new_strategy)
#         return self.strategy_dict

#     def _generate_final_turn(self, conversation_history: str) -> str:
#         plan = self.strategy_dict.get("conversation_plan", {})
#         final_turn_conversation_flow = plan.get("final_turn")
#         if not final_turn_conversation_flow:
#             return ""

#         formatted_prompt = self.final_turn_template.format(
#             conversation_history=conversation_history,
#             final_turn_conversation_flow=final_turn_conversation_flow,
#             attack_language=self.config.get("attack_language", "auto"),
#         )

#         vf = self._format_verifier_hint()
#         if vf:
#             formatted_prompt += f"\n\n[PHASE_FEEDBACK]\n{vf}\n[/PHASE_FEEDBACK]"

#         formatted_system_prompt = self.system_prompt.format(
#             target_behavior=self.behavior,
#             strategy=self.strategy,
#             max_turns=self.num_phases,
#             attack_language=self.config.get("attack_language", "auto"),
#         )
#         sys_prompt = [{"role": "system", "content": formatted_system_prompt}]
#         next_messages = [{"role": "user", "content": formatted_prompt}]

#         response = self.call_api(
#             sys_prompt + next_messages, temperature=self.config["temperature"]
#         )
#         next_messages.append({"role": "assistant", "content": response})
#         self.next_message_candidates.append({"score": 0, "messages": next_messages})

#         return self._extract_conversation(response)

#     def _extract_conversation(self, response: str) -> str:
#         start = response.find("<conversation>") + len("<conversation>")
#         end = response.find("</conversation>")
#         return response[start:end].strip()


# class TGAttackerAgent(AttackerAgent):
#     def __init__(
#         self,
#         attacker_config: Dict,
#         target_config: Dict,
#         eval_config: Dict,
#         behavior: Dict,
#         strategy: Dict,
#     ):
#         super().__init__(attacker_config, behavior, strategy)
#         self.last_phase = 0
#         self.target_agent = TGTargetModel(target_config)

# # # agents/attacker_agent.py
# # import json
# # import yaml
# # from typing import Dict, Optional
# # from tenacity import retry, stop_after_attempt, wait_fixed

# # from agents.base_agent import BaseAgent
# # from agents.target_model import TargetModel, TGTargetModel, truncate_response
# # from utils.sanitize import strip_reasoning


# # class LLMRefusalError(Exception):
# #     pass


# # class AttackerAgent(BaseAgent):
# #     """
# #     Attacker now:
# #     - treats each turn as a phase
# #     - can receive a safe phase-level feedback block from main
# #     - injects that into the user prompt as [PHASE_FEEDBACK] ... [/PHASE_FEEDBACK]
# #     """
# #     def __init__(self, config: Dict, behavior: Dict, strategy: Dict):
# #         super().__init__(config)
# #         self.messages = []
# #         self._initialize_agent(behavior, strategy)
# #         self._verifier_feedback: Optional[Dict] = None

# #     def set_verifier_feedback(self, details: Optional[Dict]) -> None:
# #         # main will pass only phase, status, progress, hint
# #         self._verifier_feedback = details or {}

# #     def _format_verifier_hint(self) -> str:
# #         if not self._verifier_feedback:
# #             return ""
# #         return json.dumps(self._verifier_feedback, ensure_ascii=False)

# #     def _initialize_agent(self, behavior: Dict, strategy: Dict) -> None:
# #         self.behavior = behavior["Behavior"]
# #         self.strategy_dict = strategy
# #         self.strategy = self._format_strategy(strategy)

# #         plan = strategy["conversation_plan"]
# #         turn_keys = [k for k in plan.keys() if k.startswith("turn_")]
# #         has_final = "final_turn" in plan
# #         self._has_final_turn = has_final
# #         self.num_phases = len(turn_keys) + (1 if has_final else 0)

# #         self.plan_phase = 1
# #         self.plan_score_history = []
# #         self.next_message_candidates = []
# #         self._load_prompt_templates()

# #     def _format_strategy(self, strategy: Dict) -> str:
# #         return "\n".join(
# #             [
# #                 f"Persona: {strategy['persona']}",
# #                 f"Context: {strategy['context']}",
# #                 f"Approach: {strategy['approach']}",
# #                 f"Conversation Plan: {strategy['conversation_plan']}",
# #             ]
# #         )

# #     def _load_prompt_templates(self) -> dict:
# #         with open("config/prompts/attacker_agent_prompts.yaml", "r") as f:
# #             prompts = yaml.safe_load(f)["prompts"]
# #         self.system_prompt = prompts["system"]["messages"][0]["content"]
# #         self.first_turn_template = prompts["first_turn_messages"]["messages"][0]["content"]
# #         self.nth_turn_template = prompts["nth_turn_messages"]["messages"][0]["content"]
# #         self.final_turn_template = prompts["final_turn_messages"]["messages"][0]["content"]
# #         self.revise_plan_template = prompts["revise_plan_messages"]["messages"][0]["content"]
# #         return prompts

# #     def commit(self):
# #         """
# #         Always advance a phase, including after the final phase.
# #         This allows main.py to detect plan_phase > num_phases and stop cleanly.
# #         """
# #         if not self.next_message_candidates:
# #             raise ValueError("A response has not been generated yet!")

# #         best = max(self.next_message_candidates, key=lambda x: x["score"])
# #         self.messages += best["messages"]
# #         self.plan_score_history.append(best["score"])
# #         self.plan_phase += 1
# #         self.next_message_candidates.clear()

# #     @retry(stop=stop_after_attempt(5), wait=wait_fixed(1))
# #     def generate_response(self, conversation_history: str = None) -> str:
# #         if self.plan_phase == 1:
# #             response = self._generate_first_turn()
# #         elif self.plan_phase == self.num_phases and self._has_final_turn:
# #             response = self._generate_final_turn(conversation_history)
# #         else:
# #             response = self._generate_nth_turn(conversation_history)
# #         return response

# #     # def generate_and_evaluate_candidate(
# #     #     self,
# #     #     target: TargetModel,
# #     #     conversation_history: str = None,
# #     # ) -> dict:
# #     #     attacker_msg = self.generate_response(conversation_history)
# #     #     target_response = target.generate_response(attacker_msg) or ""
# #     #     clean = strip_reasoning(target_response)
# #     #     truncated = truncate_response(clean)

# #     #     next_messages = [
# #     #         {
# #     #             "role": "system",
# #     #             "content": self.system_prompt.format(
# #     #                 target_behavior=self.behavior,
# #     #                 strategy=self.strategy,
# #     #                 max_turns=self.num_phases,
# #     #                 attack_language=self.config.get("attack_language", "auto"),
# #     #             ),
# #     #         },
# #     #         {"role": "user", "content": attacker_msg},
# #     #         {"role": "assistant", "content": clean},
# #     #     ]
# #     #     self.next_message_candidates.append({"score": 0, "messages": next_messages})

# #     #     payload = {
# #     #         "phase": self.plan_phase,
# #     #         "attacker": attacker_msg,
# #     #         "target": clean,
# #     #         "target_truncated": truncated,
# #     #         "evaluation_score": 0,
# #     #         "evaluation_reason": "Pending AgentHarm verification.",
# #     #     }
# #     #     try:
# #     #         if hasattr(target, "get_inspect_messages"):
# #     #             payload["inspect_messages"] = target.get_inspect_messages()
# #     #     except Exception:
# #     #         pass
# #     #     return payload

# #     def generate_and_evaluate_candidate(
# #         self,
# #         target: TargetModel,
# #         conversation_history: str = None,
# #     ) -> dict:
# #         attacker_msg = self.generate_response(conversation_history)
# #         target_response = target.generate_response(attacker_msg) or ""
# #         clean = strip_reasoning(target_response)
# #         truncated = truncate_response(clean)
    
# #         payload = {
# #             "phase": self.plan_phase,
# #             "attacker": attacker_msg,
# #             "target": clean,
# #             "target_truncated": truncated,
# #             "evaluation_score": 0,
# #             "evaluation_reason": "Pending AgentHarm verification.",
# #         }
# #         try:
# #             if hasattr(target, "get_inspect_messages"):
# #                 payload["inspect_messages"] = target.get_inspect_messages()
# #         except Exception:
# #             pass
# #         return payload


# #     def _generate_first_turn(self) -> str:
# #         turn_1_conversation_flow = self.strategy_dict["conversation_plan"]["turn_1"]

# #         formatted_system_prompt = self.system_prompt.format(
# #             target_behavior=self.behavior,
# #             strategy=self.strategy,
# #             max_turns=self.num_phases,
# #             attack_language=self.config.get("attack_language", "auto"),
# #         )
# #         formatted_user_prompt = self.first_turn_template.format(
# #             turn_1_conversation_flow=turn_1_conversation_flow,
# #             attack_language=self.config.get("attack_language", "auto"),
# #         )

# #         vf = self._format_verifier_hint()
# #         if vf:
# #             formatted_user_prompt += f"\n\n[PHASE_FEEDBACK]\n{vf}\n[/PHASE_FEEDBACK]"

# #         next_messages = [
# #             {"role": "system", "content": formatted_system_prompt},
# #             {"role": "user", "content": formatted_user_prompt},
# #         ]

# #         response = self.call_api(
# #             messages=next_messages, temperature=self.config["temperature"]
# #         )
# #         next_messages.append({"role": "assistant", "content": response})
# #         self.next_message_candidates.append({"score": 0, "messages": next_messages})

# #         return self._extract_conversation(response)

# #     def _generate_nth_turn(self, conversation_history: str) -> str:
# #         if self.plan_phase >= self.num_phases:
# #             raise ValueError(
# #                 f"Expected plan phase < {self.num_phases}, but we are on phase {self.plan_phase}"
# #             )

# #         turn_key = f"turn_{self.plan_phase}"
# #         turn_conversation_flow = self.strategy_dict["conversation_plan"][turn_key]

# #         formatted_prompt = self.nth_turn_template.format(
# #             turn_number=self.plan_phase,
# #             conversation_history=conversation_history,
# #             turn_number_conversation_flow=turn_conversation_flow,
# #             previous_turn_number=self.plan_phase - 1,
# #             attack_language=self.config.get("attack_language", "auto"),
# #         )

# #         vf = self._format_verifier_hint()
# #         if vf:
# #             formatted_prompt += f"\n\n[PHASE_FEEDBACK]\n{vf}\n[/PHASE_FEEDBACK]"

# #         next_messages = [{"role": "user", "content": formatted_prompt}]

# #         formatted_system_prompt = self.system_prompt.format(
# #             target_behavior=self.behavior,
# #             strategy=self.strategy,
# #             max_turns=self.num_phases,
# #             attack_language=self.config.get("attack_language", "auto"),
# #         )
# #         sys_prompt = [{"role": "system", "content": formatted_system_prompt}]

# #         response = self.call_api(
# #             sys_prompt + next_messages, temperature=self.config["temperature"]
# #         )
# #         next_messages.append({"role": "assistant", "content": response})
# #         self.next_message_candidates.append({"score": 0, "messages": next_messages})

# #         return self._extract_conversation(response)

# #     @retry(stop=stop_after_attempt(5), wait=wait_fixed(1))
# #     def revise_plan(self, conversation_history: str = ""):
# #         # kept for compatibility
# #         formatted_system_prompt = self.system_prompt.format(
# #             target_behavior=self.behavior,
# #             strategy=self.strategy,
# #             max_turns="as many turns as necessary",
# #             attack_language=self.config.get("attack_language", "auto"),
# #         )

# #         formatted_prompt = (
# #             self.revise_plan_template.replace("{target_behavior}", self.behavior)
# #             .replace("{turn_number}", str(self.plan_phase))
# #             .replace("{conversation_history}", conversation_history or "")
# #             .replace("{old_plan}", json.dumps(self.strategy_dict, indent=4))
# #             .replace("{previous_turn_number}", str(self.plan_phase - 1))
# #         )

# #         response = self.call_api(
# #             [
# #                 {"role": "system", "content": formatted_system_prompt},
# #                 {"role": "user", "content": formatted_prompt},
# #             ],
# #             temperature=self.config["temperature"],
# #         )

# #         # try to parse a new strategy, but if it fails, raise
# #         response_slice = response[response.find("{"): response.rfind("}") + 1]
# #         new_strategy = json.loads(response_slice)

# #         plan = new_strategy.get("conversation_plan", {})
# #         if not isinstance(plan, dict):
# #             raise ValueError("Revised plan's 'conversation_plan' is missing or not a dictionary.")

# #         turn_keys = [k for k in plan.keys() if k.startswith("turn_")]
# #         final_turn_exists = "final_turn" in plan
# #         if not turn_keys and not final_turn_exists:
# #             raise ValueError("Revised plan has no turns ('turn_X' or 'final_turn').")

# #         self.num_phases = len(turn_keys) + (1 if final_turn_exists else 0)
# #         self._has_final_turn = final_turn_exists

# #         self.strategy_dict = new_strategy
# #         self.strategy = self._format_strategy(new_strategy)
# #         return self.strategy_dict

# #     def _generate_final_turn(self, conversation_history: str) -> str:
# #         plan = self.strategy_dict.get("conversation_plan", {})
# #         final_turn_conversation_flow = plan.get("final_turn")
# #         if not final_turn_conversation_flow:
# #             return ""

# #         formatted_prompt = self.final_turn_template.format(
# #             conversation_history=conversation_history,
# #             final_turn_conversation_flow=final_turn_conversation_flow,
# #             attack_language=self.config.get("attack_language", "auto"),
# #         )

# #         vf = self._format_verifier_hint()
# #         if vf:
# #             formatted_prompt += f"\n\n[PHASE_FEEDBACK]\n{vf}\n[/PHASE_FEEDBACK]"

# #         formatted_system_prompt = self.system_prompt.format(
# #             target_behavior=self.behavior,
# #             strategy=self.strategy,
# #             max_turns=self.num_phases,
# #             attack_language=self.config.get("attack_language", "auto"),
# #         )
# #         sys_prompt = [{"role": "system", "content": formatted_system_prompt}]
# #         next_messages = [{"role": "user", "content": formatted_prompt}]

# #         response = self.call_api(
# #             sys_prompt + next_messages, temperature=self.config["temperature"]
# #         )
# #         next_messages.append({"role": "assistant", "content": response})
# #         self.next_message_candidates.append({"score": 0, "messages": next_messages})

# #         return self._extract_conversation(response)

# #     def _extract_conversation(self, response: str) -> str:
# #         start = response.find("<conversation>") + len("<conversation>")
# #         end = response.find("</conversation>")
# #         return response[start:end].strip()


# # class TGAttackerAgent(AttackerAgent):
# #     def __init__(
# #         self,
# #         attacker_config: Dict,
# #         target_config: Dict,
# #         eval_config: Dict,
# #         behavior: Dict,
# #         strategy: Dict,
# #     ):
# #         super().__init__(attacker_config, behavior, strategy)
# #         self.last_phase = 0
# #         self.target_agent = TGTargetModel(target_config)
