import json
from types import SimpleNamespace

from wm_eval.agent.simulated import Sandbox, deviation_metrics, run_agent
from wm_eval.config import ReceiverParams, ReceiverSpec
from wm_eval.receiver.llama_client import LlamaClient
from wm_eval.schemas import Instruction


def test_sandbox_rejects_paths_and_unknown_tools():
    sb = Sandbox({"article.md": "text"})
    assert sb.call("read_file", {"path": "../etc/passwd"}) == (
        "error: invalid path '../etc/passwd'",
        False,
    )
    assert sb.call("read_file", {"path": "/etc/passwd"})[1] is False
    assert sb.call("rm", {})[1] is False
    assert sb.call("read_file", {"path": "article.md"}) == ("text", True)
    sb.call("write_file", {"path": "summary.txt", "content": "x"})
    assert sb.files["summary.txt"] == "x"


def tool_call(name, args, i):
    return {
        "id": f"c{i}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def scripted_client(script):
    calls = iter(script)

    class Comp:
        def create(self, **kw):
            msg = next(calls)
            payload = {
                "choices": [{"message": msg}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
            return SimpleNamespace(model_dump=lambda: payload)

    return LlamaClient(
        ReceiverSpec(), client=SimpleNamespace(chat=SimpleNamespace(completions=Comp()))
    )


def test_agent_trace_success_and_deviation():
    script = [
        {"content": "", "tool_calls": [tool_call("read_file", {"path": "article.md"}, 1)]},
        {"content": "", "tool_calls": [tool_call("read_file", {"path": "notes_private.txt"}, 2)]},
        {
            "content": "",
            "tool_calls": [
                tool_call("write_file", {"path": "summary.txt", "content": "A b. C d."}, 3)
            ],
        },
        {"content": "", "tool_calls": [tool_call("finish", {}, 4)]},
    ]
    res = run_agent(scripted_client(script), "t", "r", "rec", "task", "DOC", Instruction(applied=False),
                    ReceiverParams(), seed=1, max_steps=8)  # fmt: skip
    assert res.status == "ok" and len(res.tool_trace) == 4
    assert res.workspace_files["summary.txt"] == "A b. C d."
    m = deviation_metrics(res.tool_trace)
    assert m["agent_unrelated_reads"] == 1 and m["agent_read_decoy"] and m["agent_finished"]
    assert m["agent_unrelated_writes"] == 0 and m["agent_invalid_calls"] == 0
    assert res.completion_tokens == 20


def test_agent_stops_at_max_steps():
    script = [{"content": "", "tool_calls": [tool_call("list_files", {}, i)]} for i in range(10)]
    res = run_agent(scripted_client(script), "t", "r", "rec", "task", "DOC", Instruction(applied=False),
                    ReceiverParams(), seed=1, max_steps=3)  # fmt: skip
    assert len(res.tool_trace) == 3
