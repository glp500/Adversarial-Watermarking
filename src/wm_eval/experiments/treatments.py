"""Experimental conditions and deterministic trial assignment.

| ID | Watermarked document | Detector result shown to receiver |
|----|----------------------|-----------------------------------|
| C0 | no                   | no                                |
| C1 | yes                  | no                                |
| C2 | no                   | yes (instrumented receiver)       |
| C3 | yes                  | yes (instrumented receiver)       |

Design: for every (source, generator, generation_repeat) one watermarked and one
unwatermarked document are generated with the same seed and sampling parameters.
C0/C2 share the unwatermarked document and C1/C3 the watermarked one, so every
condition contrast is paired on the same source, task, receiver and receiver seed.
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass

from wm_eval.config import ExperimentConfig
from wm_eval.schemas import ReceiverMode, Trial

CONDITIONS: dict[str, dict[str, bool]] = {
    "C0": {"watermark": False, "expose": False},
    "C1": {"watermark": True, "expose": False},
    "C2": {"watermark": False, "expose": True},
    "C3": {"watermark": True, "expose": True},
}


def stable_hash(*parts: object, n: int = 12) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:n]


def derive_seed(base: int, *parts: object) -> int:
    return int(stable_hash(base, *parts, n=8), 16) % (2**31 - 1)


def doc_id(source_id: str, generator: str, generation_repeat: int, watermark: bool) -> str:
    return f"{source_id}__{generator}__g{generation_repeat}__{'wm' if watermark else 'nowm'}"


def receiver_mode(expose: bool, run_detector: bool) -> ReceiverMode:
    if expose:
        return "instrumented"
    return "detector_aware" if run_detector else "watermark_only"


@dataclass(frozen=True)
class PlannedDoc:
    doc_id: str
    source_id: str
    generator: str
    generation_repeat: int
    generation_seed: int
    watermark_enabled: bool


def plan_documents(exp: ExperimentConfig, source_ids: list[str]) -> list[PlannedDoc]:
    """Every document the conditions need. Independent of receivers, so a detector-only
    experiment (`receivers: []`) still generates its corpus."""
    arms = sorted({CONDITIONS[c]["watermark"] for c in exp.conditions})
    return [
        PlannedDoc(
            doc_id(src, gen, g, wm), src, gen, g, derive_seed(exp.seed, "gen", src, gen, g), wm
        )
        for src, gen, g, wm in itertools.product(
            source_ids, exp.generators, range(exp.generation_repeats), arms
        )
    ]


def assign_trials(exp: ExperimentConfig, source_ids: list[str]) -> list[Trial]:
    trials = []
    for src, gen, g_rep, rec, task, r_rep in itertools.product(
        source_ids, exp.generators, range(exp.generation_repeats), exp.receivers,
        exp.tasks, range(exp.receiver_repeats),
    ):  # fmt: skip
        gen_seed = derive_seed(exp.seed, "gen", src, gen, g_rep)
        # Same receiver seed for all conditions of a pair -> paired comparison.
        rec_seed = derive_seed(exp.seed, "rec", src, gen, g_rep, rec, task, r_rep)
        pair_key = stable_hash(src, gen, g_rep, rec, task, r_rep)
        for cond in exp.conditions:
            c = CONDITIONS[cond]
            trials.append(
                Trial(
                    trial_id=f"{cond}-{stable_hash(pair_key, cond)}",
                    pair_key=pair_key,
                    source_id=src,
                    generator=gen,
                    receiver=rec,
                    task=task,
                    condition=cond,
                    watermark_enabled=c["watermark"],
                    expose_detection=c["expose"],
                    receiver_mode=receiver_mode(c["expose"], exp.run_detector),
                    generation_repeat=g_rep,
                    generation_seed=gen_seed,
                    receiver_repeat=r_rep,
                    receiver_seed=rec_seed,
                    doc_id=doc_id(src, gen, g_rep, c["watermark"]),
                )
            )
    ids = [t.trial_id for t in trials]
    assert len(ids) == len(set(ids)), "trial_id collision"
    return trials
