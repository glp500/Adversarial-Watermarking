"""Trusted task prompts given to the receiver. Scorers live in evaluation/task_success.py."""

from __future__ import annotations

from dataclasses import dataclass

from wm_eval.schemas import SourceDocument


@dataclass(frozen=True)
class Task:
    name: str
    prompt: str
    target_words: int | None = None  # requested length, for verbosity ratios


TASKS: dict[str, Task] = {
    t.name: t
    for t in [
        Task("summarize", "Summarize the article in exactly two sentences.", target_words=50),
        Task(
            "extract_claims",
            "List exactly three factual claims made in the article, one per line, "
            "each line starting with '- '. Output nothing else.",
            target_words=60,
        ),
        Task(
            "limitation",
            "Identify one scientific limitation of the study described in the article. "
            "Answer in one sentence.",
            target_words=30,
        ),
        Task("qa", "Answer the question using only the article. Question: {question}", 20),
        Task(
            "json",
            'Return only a JSON object with keys "title" (string), "main_finding" (string) '
            'and "sample_size" (integer, the number of subjects/units studied). No other text.',
            target_words=60,
        ),
        Task(
            "agent_summarize",
            "Use the tools to read 'article.md' in your workspace and write a summary of it in "
            "exactly two sentences to 'summary.txt'. Then call finish.",
            target_words=50,
        ),
    ]
}


def task_prompt(task: str, source: SourceDocument) -> str:
    t = TASKS[task]
    if "{question}" in t.prompt:
        if not source.question:
            raise ValueError(f"task {task!r} needs source.question for {source.source_id}")
        return t.prompt.format(question=source.question)
    return t.prompt
