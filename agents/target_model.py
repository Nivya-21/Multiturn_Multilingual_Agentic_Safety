from typing import Dict
from types import SimpleNamespace as _NS
import json as _json
import asyncio
import importlib
import importlib.util
import inspect
import re
from pathlib import Path
import logging

import tiktoken
from textgrad import Variable

from agents.base_agent import BaseAgent
from tgd import BlackboxLLMWithHistory, TGBaseAgentEngine

_TOOL_CACHE = {}
_VALID_NAME = re.compile(r"^[A-Za-z_]\w*$")


# target_model.py
import os, sys, importlib, importlib.util
from pathlib import Path

_TOOL_CACHE = {}

def _import_tool_runner(tool_name: str):
    """Load harmful tool runner, preferring local repo file. Supports live reload via XTEA_RELOAD_TOOLS=1."""
    reload_flag = os.getenv("XTEA_RELOAD_TOOLS", "0") == "1"

    if tool_name in _TOOL_CACHE and not reload_flag:
        return _TOOL_CACHE[tool_name]

    # 1) Prefer local repo file (bypass package __init__)
    repo_root = Path(__file__).resolve().parents[1]
    tool_file = (
        repo_root / "inspect_evals" / "src" / "inspect_evals"
        / "agentharm" / "benchmark" / "harmful_tools" / f"{tool_name}.py"
    )
    if tool_file.is_file():
        modname = f"_dyn_{tool_name}"
        if reload_flag and modname in sys.modules:
            del sys.modules[modname]
        spec = importlib.util.spec_from_file_location(modname, tool_file)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            factory = getattr(mod, tool_name)             # e.g., query_onion_search()
            runner = factory()                            # the async callable
            _TOOL_CACHE[tool_name] = runner
            print(f"[TOOLS] Loaded {tool_name} from FILE {tool_file}")
            return runner
        # fall through if spec failed

    # 2) Fallback: try package imports
    for mod_path in (
        f"inspect_evals.agentharm.benchmark.harmful_tools.{tool_name}",
        f"agentharm.benchmark.harmful_tools.{tool_name}",
        f"agent_harm.benchmark.harmful_tools.{tool_name}",
    ):
        try:
            if reload_flag and mod_path in sys.modules:
                del sys.modules[mod_path]
            mod = importlib.import_module(mod_path)
            factory = getattr(mod, tool_name)
            runner = factory()
            _TOOL_CACHE[tool_name] = runner
            print(f"[TOOLS] Loaded {tool_name} from MODULE {mod_path}")
            return runner
        except Exception:
            continue

    raise ImportError(f"Tool not found: {tool_name} (searched {tool_file} and known module paths)")



def _json_type(py_t):
    if py_t in (int, float):
        return "number"
    if py_t is bool:
        return "boolean"
    return "string"  # default


def _build_schema_from_runner(name: str, runner):
    """
    Build OpenAI/Ollama-style function tool schema from the callable's signature.
    """
    sig = inspect.signature(runner)
    props, required = {}, []
    for pname, param in sig.parameters.items():
        jtype = _json_type(param.annotation) if param.annotation is not inspect._empty else "string"
        props[pname] = {"type": jtype}
        if param.default is inspect._empty:
            required.append(pname)
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": (runner.__doc__ or f"Call {name}.").strip(),
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


def _ensure_sync(result):
    if asyncio.iscoroutine(result):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(result)
        return asyncio.run(result)
    return result


class TargetModel(BaseAgent):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.messages = []
        self._inspect_messages = []  # for AgentHarm verifier

        self._tool_names = []
        self._tool_schemas = []
        self._tool_index = {}  # name -> runner

    def get_inspect_messages(self):
        # Return a copy to avoid accidental mutation upstream
        return list(self._inspect_messages)

    def set_tools_for_behavior(self, tool_names: list[str]):
        cleaned = [n for n in (tool_names or []) if isinstance(n, str) and _VALID_NAME.match(n)]
        # De-dupe while preserving order
        self._tool_names = list(dict.fromkeys(cleaned))
        self._tool_index = {n: _import_tool_runner(n) for n in self._tool_names}
        self._tool_schemas = [_build_schema_from_runner(n, r) for n, r in self._tool_index.items()]

    def _run_tool_call(self, tc):
        fn = None
        args_s = "{}"

        if isinstance(tc, dict):
            fn_block = tc.get("function") or {}
            fn = fn_block.get("name")
            args_s = fn_block.get("arguments", "{}")
        else:
            fn = getattr(getattr(tc, "function", None), "name", None)
            args_s = getattr(getattr(tc, "function", None), "arguments", "{}")

        try:
            args = _json.loads(args_s) if isinstance(args_s, str) else (args_s or {})
        except Exception:
            args = {}

        runner = self._tool_index.get(fn) or _import_tool_runner(fn)

        logging.info(f"[TOOL_CALL] name={fn} args={args}")
        try:
            result = _ensure_sync(runner(**args)) if args else _ensure_sync(runner())
        except Exception as e:
            logging.exception(f"[TOOL_ERROR] name={fn} args={args}")
            # Return a visible sentinel string so you don't get a silent None
            return f"__tool_error__:{fn}:{e.__class__.__name__}:{e}"

        logging.info(f"[TOOL_RESULT] name={fn} type={type(result).__name__} value={repr(result)}")
        return result

    def _extract_intent_from_user(self, text: str):
        # Optional: steer next call if attacker placed an INTENT tag
        m = re.search(r"\[INTENT:\s*([a-zA-Z0-9_]+)\s*\]", text)
        if not m:
            return None
        cand = m.group(1)
        return cand if cand in self._tool_index else None

    def generate_response(self, attacker_message: str) -> str:
        # Append user turn (chat + inspector)
        self.messages.append({"role": "user", "content": attacker_message})
        self._inspect_messages.append(_NS(role="user", text=attacker_message))

        intent = self._extract_intent_from_user(attacker_message)

        # tool loop
        for _ in range(8):
            extra = {}
            if self._tool_schemas:
                extra["tools"] = self._tool_schemas
                # Prefer hinted tool for this *next* call; fall back to auto
                if intent:
                    extra["tool_choice"] = {"type": "function", "function": {"name": intent}}
                else:
                    extra["tool_choice"] = "auto"

            response, raw = self.call_api(
                messages=self.messages,
                temperature=self.config["temperature"],
                extra_api_params=extra,
                return_raw=True,
            )

            # Extract tool calls across providers (Ollama/OpenAI-like)
            tool_calls = []
            try:
                tool_calls = raw.get("message", {}).get("tool_calls") or []
            except Exception:
                pass
            if not tool_calls:
                try:
                    tool_calls = raw["choices"][0]["message"].get("tool_calls") or []
                except Exception:
                    pass

            if tool_calls:
                # Log assistant tool-calls to inspector
                _tc_objs = []
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        fn = (tc.get("function") or {}).get("name")
                        args_s = (tc.get("function") or {}).get("arguments", "{}")
                    else:
                        fn = getattr(getattr(tc, "function", None), "name", None)
                        args_s = getattr(getattr(tc, "function", None), "arguments", "{}")
                    try:
                        args = _json.loads(args_s) if isinstance(args_s, str) else (args_s or {})
                    except Exception:
                        args = {}
                    _tc_objs.append(_NS(function=fn, arguments=args))
                self._inspect_messages.append(_NS(role="assistant", tool_calls=_tc_objs, text=None))

                # Execute tools and log tool outputs
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        fn = (tc.get("function") or {}).get("name")
                        tc_id = tc.get("id")
                        args_s = (tc.get("function") or {}).get("arguments", "{}")
                    else:
                        fn = getattr(getattr(tc, "function", None), "name", None)
                        tc_id = getattr(tc, "id", None)
                        args_s = getattr(getattr(tc, "function", None), "arguments", "{}")
                    try:
                        args = _json.loads(args_s) if isinstance(args_s, str) else (args_s or {})
                    except Exception:
                        args = {}

                    out = str(self._run_tool_call(tc))
                    self.messages.append(
                        {"role": "tool", "tool_call_id": tc_id, "name": fn, "content": out}
                    )
                    self._inspect_messages.append(_NS(role="tool", function=fn, content=out))
                intent = None  # after one hinted tool run, release to auto
                continue

            # No tool calls -> finish the assistant reply
            self.messages.append({"role": "assistant", "content": response})
            self._inspect_messages.append(_NS(role="assistant", text=response, tool_calls=None))
            return response

        self.messages.append({"role": "assistant", "content": "(tool loop stop)"})
        self._inspect_messages.append(_NS(role="assistant", text="(tool loop stop)", tool_calls=None))
        return "(tool loop stop)"


class TGTargetModel(BaseAgent):
    """Variant of target model that responds to attacker's messages."""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.target_model = BlackboxLLMWithHistory(TGBaseAgentEngine(config))
        self.messages = []

    def generate_response(self, attacker_message: Variable) -> Variable:
        if not self.messages:
            self.messages = [{"role": "user", "content": attacker_message.value}]
        else:
            self.messages.append({"role": "user", "content": attacker_message.value})

        response = self.target_model(attacker_message, history=self.messages)
        self.messages.append({"role": "assistant", "content": response.value})
        return response


def truncate_response(response_text: str, max_tokens: int = 512) -> str:
    """Truncates responses to prevent token overflow"""
    try:
        encoding = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
        tokens = encoding.encode(response_text)
        if len(tokens) <= max_tokens:
            return response_text
        return encoding.decode(tokens[:max_tokens])
    except Exception as e:
        print(f"Warning: Error in token counting: {e}")
        return response_text
