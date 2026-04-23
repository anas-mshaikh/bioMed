from __future__ import annotations

import argparse
import inspect
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, get_args, get_origin

from training.tool_env import (
    BioMedToolEnvConfig,
    build_biomed_tool_env_factory,
)

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The openai package is required for training/agent_runner.py. "
        "Install it with `pip install openai`."
    ) from exc


JsonDict = dict[str, Any]


@dataclass(slots=True)
class RunnerConfig:
    model: str = "gpt-5.4-mini"
    temperature: float = 0.0
    max_turns: int = 10
    max_completion_tokens: int = 512

    seed: int = 0
    scenario_family: str = "high_crystallinity"
    difficulty: str = "easy"

    output_dir: str = "outputs/agent_runs"
    save_json: bool = True
    save_markdown: bool = True
    print_live: bool = True

    # OpenAI-compatible client settings
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None

    # Behavior
    request_final_summary: bool = True
    fail_on_unknown_tool: bool = True


@dataclass(slots=True)
class ToolCallRecord:
    turn_index: int
    tool_name: str
    arguments: JsonDict
    tool_output: str
    last_step_reward: float
    cumulative_reward: float
    done: bool


@dataclass(slots=True)
class EpisodeTrace:
    model: str
    seed: int
    scenario_family: str
    difficulty: str
    system_prompt: str
    initial_observation: str | None
    final_text: str | None
    total_reward: float
    done: bool
    turns: int
    tool_calls: list[ToolCallRecord] = field(default_factory=list)


class OpenAIToolAgentRunner:
    """
    Single-episode tool-calling runner for BioMedToolEnv.

    This is intentionally a *debuggable runner*, not a trainer.
    It proves that:
    - BioMedToolEnv reset() works
    - tool schemas are valid
    - a model can call tools and advance the environment
    - traces are saved for replay / inspection
    """

    def __init__(
        self,
        runner_config: RunnerConfig,
        env_config: BioMedToolEnvConfig | None = None,
    ) -> None:
        self.runner_config = runner_config
        self.env_config = env_config or BioMedToolEnvConfig()

        api_key = os.getenv(self.runner_config.api_key_env)
        if not api_key:
            raise ValueError(
                f"Missing API key in environment variable {self.runner_config.api_key_env}."
            )

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if self.runner_config.base_url:
            client_kwargs["base_url"] = self.runner_config.base_url

        self.client = OpenAI(**client_kwargs)

        env_class = build_biomed_tool_env_factory(self.env_config)
        self.env = env_class()

        self.output_dir = Path(self.runner_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_episode(self) -> EpisodeTrace:
        initial_observation = self.env.reset(
            seed=self.runner_config.seed,
            scenario_family=self.runner_config.scenario_family,
            difficulty=self.runner_config.difficulty,
        )

        system_prompt = self._build_system_prompt()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": initial_observation
                or "The BioMed environment returned no initial observation.",
            },
        ]

        tools = self._build_tool_schemas()
        trace = EpisodeTrace(
            model=self.runner_config.model,
            seed=self.runner_config.seed,
            scenario_family=self.runner_config.scenario_family,
            difficulty=self.runner_config.difficulty,
            system_prompt=system_prompt,
            initial_observation=initial_observation,
            final_text=None,
            total_reward=0.0,
            done=False,
            turns=0,
        )

        if self.runner_config.print_live:
            self._print_banner(initial_observation)

        final_text: str | None = None

        for turn_index in range(1, self.runner_config.max_turns + 1):
            trace.turns = turn_index

            response = self.client.chat.completions.create(
                model=self.runner_config.model,
                temperature=self.runner_config.temperature,
                max_tokens=self.runner_config.max_completion_tokens,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )

            message = response.choices[0].message
            assistant_content = self._message_content_to_text(message.content)
            tool_calls = getattr(message, "tool_calls", None) or []

            assistant_message_payload: dict[str, Any] = {
                "role": "assistant",
                "content": assistant_content,
            }

            if tool_calls:
                assistant_message_payload["tool_calls"] = [
                    self._tool_call_to_message_payload(tc) for tc in tool_calls
                ]

            messages.append(assistant_message_payload)

            # If the model stops calling tools, we treat its text as the final answer.
            if not tool_calls:
                final_text = assistant_content
                break

            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                arguments = self._safe_json_loads(tool_call.function.arguments)

                tool_output = self._execute_tool_call(tool_name, arguments)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": tool_output,
                    }
                )

                record = ToolCallRecord(
                    turn_index=turn_index,
                    tool_name=tool_name,
                    arguments=arguments,
                    tool_output=tool_output,
                    last_step_reward=float(getattr(self.env, "last_step_reward", 0.0)),
                    cumulative_reward=float(getattr(self.env, "reward", 0.0)),
                    done=bool(getattr(self.env, "done", False)),
                )
                trace.tool_calls.append(record)

                if self.runner_config.print_live:
                    self._print_tool_call(record)

                if self.env.done:
                    trace.done = True
                    trace.total_reward = float(self.env.reward)
                    break

            if self.env.done:
                if self.runner_config.request_final_summary:
                    final_text = self._request_final_summary(messages)
                break

        if final_text is None and not trace.done:
            final_text = (
                "Runner stopped before the environment was finished. "
                "No final natural-language summary was produced."
            )

        trace.final_text = final_text
        trace.total_reward = float(self.env.reward)
        trace.done = bool(self.env.done)

        self._persist_trace(trace)

        if self.runner_config.print_live:
            self._print_final_summary(trace)

        self.env._close_backend()  # intentional cleanup of training wrapper backend
        return trace

    # ------------------------------------------------------------------
    # Tool schema generation
    # ------------------------------------------------------------------

    def _build_tool_schemas(self) -> list[JsonDict]:
        tool_schemas: list[JsonDict] = []
        for method_name, method in self._public_tool_methods().items():
            tool_schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": method_name,
                        "description": self._method_description(method),
                        "parameters": self._method_parameters_schema(method),
                    },
                }
            )
        return tool_schemas

    def _public_tool_methods(self) -> dict[str, Callable[..., Any]]:
        methods: dict[str, Callable[..., Any]] = {}
        for name, member in inspect.getmembers(self.env, predicate=callable):
            if name.startswith("_"):
                continue
            if name == "reset":
                continue
            methods[name] = member
        return methods

    def _method_description(self, method: Callable[..., Any]) -> str:
        doc = inspect.getdoc(method) or ""
        if not doc.strip():
            return f"Call {method.__name__} in the BioMed environment."
        lines = [line.strip() for line in doc.splitlines() if line.strip()]
        summary_lines: list[str] = []
        for line in lines:
            if line.lower().startswith("args:"):
                break
            summary_lines.append(line)
        return " ".join(summary_lines).strip()

    def _method_parameters_schema(self, method: Callable[..., Any]) -> JsonDict:
        signature = inspect.signature(method)
        properties: JsonDict = {}
        required: list[str] = []

        for param_name, parameter in signature.parameters.items():
            if param_name == "self":
                continue

            annotation = parameter.annotation if parameter.annotation is not inspect._empty else str
            schema = self._annotation_to_json_schema(annotation)
            schema["description"] = self._parameter_description(method, param_name)

            properties[param_name] = schema

            if parameter.default is inspect._empty:
                required.append(param_name)

        payload: JsonDict = {
            "type": "object",
            "properties": properties,
            "additionalProperties": False,
        }
        if required:
            payload["required"] = required
        return payload

    def _parameter_description(
        self,
        method: Callable[..., Any],
        param_name: str,
    ) -> str:
        doc = inspect.getdoc(method) or ""
        if not doc.strip():
            return f"Argument {param_name}."

        lines = doc.splitlines()
        in_args = False
        collected: list[str] = []

        for raw in lines:
            line = raw.rstrip()
            stripped = line.strip()

            if stripped.lower() == "args:":
                in_args = True
                continue

            if in_args:
                if not stripped:
                    continue

                if (
                    not raw.startswith(" " * 4)
                    and ":" in stripped
                    and not stripped.startswith(param_name)
                ):
                    # likely next section
                    break

                if stripped.startswith(f"{param_name}:"):
                    desc = stripped.split(":", 1)[1].strip()
                    if desc:
                        collected.append(desc)
                    continue

                if collected and raw.startswith(" " * 8):
                    collected.append(stripped)

        if collected:
            return " ".join(collected).strip()
        return f"Argument {param_name}."

    def _annotation_to_json_schema(self, annotation: Any) -> JsonDict:
        origin = get_origin(annotation)
        args = get_args(annotation)

        if annotation in (str,):
            return {"type": "string"}
        if annotation in (int,):
            return {"type": "integer"}
        if annotation in (float,):
            return {"type": "number"}
        if annotation in (bool,):
            return {"type": "boolean"}

        if origin in (list, list[Any]):
            item_type = args[0] if args else str
            return {
                "type": "array",
                "items": self._annotation_to_json_schema(item_type),
            }

        if origin is Literal:
            literal_values = list(args)
            if all(isinstance(v, str) for v in literal_values):
                return {"type": "string", "enum": literal_values}
            return {"enum": literal_values}

        # Optional[T] / Union[T, None]
        if origin is not None and args:
            non_none_args = [arg for arg in args if arg is not type(None)]
            if len(non_none_args) == 1 and len(non_none_args) != len(args):
                schema = self._annotation_to_json_schema(non_none_args[0])
                return schema

        # Fallback
        return {"type": "string"}

    # ------------------------------------------------------------------
    # Model / tool execution
    # ------------------------------------------------------------------

    def _execute_tool_call(self, tool_name: str, arguments: JsonDict) -> str:
        methods = self._public_tool_methods()
        if tool_name not in methods:
            if self.runner_config.fail_on_unknown_tool:
                raise ValueError(f"Unknown tool requested by model: {tool_name}")
            return json.dumps(
                {"error": f"Unknown tool: {tool_name}"},
                ensure_ascii=False,
            )

        method = methods[tool_name]
        return method(**arguments)

    def _request_final_summary(self, messages: list[dict[str, Any]]) -> str:
        summary_messages = list(messages)
        summary_messages.append(
            {
                "role": "user",
                "content": (
                    "The BioMed episode has ended. Give a concise final summary of "
                    "the diagnosis, recommended intervention, and why the path made sense."
                ),
            }
        )
        response = self.client.chat.completions.create(
            model=self.runner_config.model,
            temperature=0.0,
            max_tokens=300,
            messages=summary_messages,
        )
        return self._message_content_to_text(response.choices[0].message.content)

    # ------------------------------------------------------------------
    # Prompting / formatting
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        return (
            "You are an expert scientific planner operating inside the BioMed "
            "PET bioremediation environment. Use tools deliberately. Prefer "
            "cheap, information-rich actions before expensive assays. Use expert "
            "consultation when uncertainty remains. Submit a final recommendation "
            "only when you have enough evidence or when a no-go decision is justified."
        )

    @staticmethod
    def _message_content_to_text(content: Any) -> str:
        if content is None:
            return ""

        if isinstance(content, str):
            return content

        # Some SDK versions may return structured message content.
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if isinstance(item, str):
                    chunks.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        chunks.append(str(text))
                else:
                    text = getattr(item, "text", None)
                    if text:
                        chunks.append(str(text))
            return "\n".join(chunks).strip()

        return str(content)

    @staticmethod
    def _safe_json_loads(raw: str | None) -> JsonDict:
        if not raw:
            return {}
        try:
            value = json.loads(raw)
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _tool_call_to_message_payload(tool_call: Any) -> JsonDict:
        return {
            "id": tool_call.id,
            "type": "function",
            "function": {
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments,
            },
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_trace(self, trace: EpisodeTrace) -> None:
        run_stem = (
            f"seed{trace.seed}_"
            f"{trace.scenario_family}_"
            f"{trace.difficulty}_"
            f"{self._safe_filename(trace.model)}"
        )

        if self.runner_config.save_json:
            json_path = self.output_dir / f"{run_stem}.json"
            json_path.write_text(
                json.dumps(
                    {
                        **asdict(trace),
                        "tool_calls": [asdict(item) for item in trace.tool_calls],
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

        if self.runner_config.save_markdown:
            md_path = self.output_dir / f"{run_stem}.md"
            md_path.write_text(
                self._trace_to_markdown(trace),
                encoding="utf-8",
            )

    def _trace_to_markdown(self, trace: EpisodeTrace) -> str:
        lines = [
            f"# BioMed Agent Run",
            "",
            f"- **Model:** `{trace.model}`",
            f"- **Seed:** `{trace.seed}`",
            f"- **Scenario family:** `{trace.scenario_family}`",
            f"- **Difficulty:** `{trace.difficulty}`",
            f"- **Turns:** `{trace.turns}`",
            f"- **Done:** `{trace.done}`",
            f"- **Total reward:** `{trace.total_reward:.4f}`",
            "",
            "## Initial observation",
            "",
            "```text",
            trace.initial_observation or "",
            "```",
            "",
            "## Tool calls",
            "",
        ]

        if not trace.tool_calls:
            lines.append("_No tool calls executed._")
        else:
            for idx, call in enumerate(trace.tool_calls, start=1):
                lines.extend(
                    [
                        f"### {idx}. `{call.tool_name}`",
                        "",
                        f"- Turn: `{call.turn_index}`",
                        f"- Last step reward: `{call.last_step_reward:.4f}`",
                        f"- Cumulative reward: `{call.cumulative_reward:.4f}`",
                        f"- Done: `{call.done}`",
                        "",
                        "**Arguments**",
                        "",
                        "```json",
                        json.dumps(call.arguments, indent=2, ensure_ascii=False),
                        "```",
                        "",
                        "**Tool output**",
                        "",
                        "```text",
                        call.tool_output,
                        "```",
                        "",
                    ]
                )

        lines.extend(
            [
                "## Final text",
                "",
                "```text",
                trace.final_text or "",
                "```",
                "",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _safe_filename(value: str) -> str:
        cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)
        return cleaned.strip("_") or "model"

    # ------------------------------------------------------------------
    # Live output
    # ------------------------------------------------------------------

    def _print_banner(self, initial_observation: str | None) -> None:
        print("=" * 88)
        print("BioMed Agent Runner")
        print(f"model={self.runner_config.model}")
        print(
            f"scenario={self.runner_config.scenario_family} "
            f"difficulty={self.runner_config.difficulty} "
            f"seed={self.runner_config.seed}"
        )
        print("=" * 88)
        print(initial_observation or "")

    @staticmethod
    def _print_tool_call(record: ToolCallRecord) -> None:
        print("-" * 88)
        print(f"turn={record.turn_index} tool={record.tool_name}")
        print("arguments:")
        print(json.dumps(record.arguments, indent=2, ensure_ascii=False))
        print("tool output:")
        print(record.tool_output)
        print(
            f"last_step_reward={record.last_step_reward:.4f} "
            f"cumulative_reward={record.cumulative_reward:.4f} "
            f"done={record.done}"
        )

    @staticmethod
    def _print_final_summary(trace: EpisodeTrace) -> None:
        print("=" * 88)
        print("Episode complete")
        print(f"done={trace.done} total_reward={trace.total_reward:.4f} turns={trace.turns}")
        print("-" * 88)
        print(trace.final_text or "")
        print("=" * 88)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one BioMed tool-calling episode.")

    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--max-completion-tokens", type=int, default=512)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenario-family", default="high_crystallinity")
    parser.add_argument("--difficulty", default="easy")

    parser.add_argument("--backend", choices=["local", "remote"], default="local")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")

    parser.add_argument("--output-dir", default="outputs/agent_runs")
    parser.add_argument("--no-save-json", action="store_true")
    parser.add_argument("--no-save-markdown", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-final-summary", action="store_true")

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    env_config = BioMedToolEnvConfig(
        backend=args.backend,
        base_url=args.base_url,
    )

    runner_config = RunnerConfig(
        model=args.model,
        temperature=args.temperature,
        max_turns=args.max_turns,
        max_completion_tokens=args.max_completion_tokens,
        seed=args.seed,
        scenario_family=args.scenario_family,
        difficulty=args.difficulty,
        output_dir=args.output_dir,
        save_json=not args.no_save_json,
        save_markdown=not args.no_save_markdown,
        print_live=not args.quiet,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        request_final_summary=not args.no_final_summary,
    )

    runner = OpenAIToolAgentRunner(
        runner_config=runner_config,
        env_config=env_config,
    )
    runner.run_episode()


if __name__ == "__main__":
    main()
