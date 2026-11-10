from types import SimpleNamespace

from wm_eval.config import ReceiverParams, ReceiverSpec
from wm_eval.receiver.inference import run_receiver
from wm_eval.receiver.llama_client import LlamaClient


class FakeCompletions:
    def __init__(self, fail=False):
        self.fail, self.calls = fail, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise TimeoutError("server timed out")
        payload = {"choices": [{"message": {"content": "Two. Sentences.", "reasoning_content": None}}],
                   "usage": {"prompt_tokens": 50, "completion_tokens": 4}, "model": "m"}  # fmt: skip
        return SimpleNamespace(model_dump=lambda: payload)


def client(fail=False, **spec):
    comp = FakeCompletions(fail)
    fake = SimpleNamespace(chat=SimpleNamespace(completions=comp))
    return LlamaClient(ReceiverSpec(**spec), client=fake), comp


def test_success_records_request_response_tokens_timing():
    c, comp = client(chat_template_kwargs={"enable_thinking": False})
    res = run_receiver(
        c, "t1", "r1", "rec", [{"role": "user", "content": "hi"}], ReceiverParams(), seed=5
    )
    assert res.status == "ok" and res.text == "Two. Sentences."
    assert res.prompt_tokens == 50 and res.completion_tokens == 4 and res.duration_s >= 0
    assert res.request["seed"] == 5 and res.request["chat_template_kwargs"] == {
        "enable_thinking": False
    }
    assert comp.calls[0]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert res.response["usage"]["completion_tokens"] == 4


def test_failure_is_returned_not_raised():
    c, _ = client(fail=True)
    res = run_receiver(
        c, "t1", "r1", "rec", [{"role": "user", "content": "hi"}], ReceiverParams(), seed=5
    )
    assert res.status == "failed" and "TimeoutError" in res.error and res.request["messages"]


def test_health_never_raises_when_server_down():
    c, _ = client(base_url="http://127.0.0.1:9/v1")
    h = c.health(timeout=0.5)
    assert h["ok"] is False and "error" in h
