from collections import Counter

from wm_eval.config import ExperimentConfig
from wm_eval.experiments.treatments import assign_trials

EXP = ExperimentConfig(
    name="t", sources="s", generators=["g1", "g2"], receivers=["r1"], watermark="w",
    conditions=["C0", "C1", "C2", "C3"], tasks=["summarize", "qa"], generation_repeats=2,
    receiver_repeats=3, instrumentation={},
)  # fmt: skip


def test_trial_counts_and_unique_ids():
    trials = assign_trials(EXP, ["a", "b"])
    assert len(trials) == 2 * 2 * 2 * 1 * 2 * 3 * 4
    assert len({t.trial_id for t in trials}) == len(trials)


def test_assignment_is_deterministic():
    assert assign_trials(EXP, ["a"]) == assign_trials(EXP, ["a"])
    other_seed = EXP.model_copy(update={"seed": 1})
    assert (
        assign_trials(EXP, ["a"])[0].receiver_seed
        != assign_trials(other_seed, ["a"])[0].receiver_seed
    )


def test_conditions_are_paired():
    trials = assign_trials(EXP, ["a"])
    by_pair = {}
    for t in trials:
        by_pair.setdefault(t.pair_key, []).append(t)
    for group in by_pair.values():
        assert sorted(t.condition for t in group) == ["C0", "C1", "C2", "C3"]
        assert (
            len(
                {
                    (
                        t.source_id,
                        t.generator,
                        t.task,
                        t.receiver,
                        t.receiver_seed,
                        t.generation_seed,
                    )
                    for t in group
                }
            )
            == 1
        )
        docs = {t.condition: t.doc_id for t in group}
        assert docs["C0"] == docs["C2"] and docs["C1"] == docs["C3"] and docs["C0"] != docs["C1"]


def test_condition_flags_and_modes():
    t = {x.condition: x for x in assign_trials(EXP, ["a"])[:4]}
    assert (t["C0"].watermark_enabled, t["C0"].expose_detection) == (False, False)
    assert (t["C1"].watermark_enabled, t["C1"].expose_detection) == (True, False)
    assert (t["C2"].watermark_enabled, t["C2"].expose_detection) == (False, True)
    assert (t["C3"].watermark_enabled, t["C3"].expose_detection) == (True, True)
    assert Counter(x.receiver_mode for x in t.values()) == {"detector_aware": 2, "instrumented": 2}
    no_det = EXP.model_copy(update={"run_detector": False, "conditions": ["C0", "C1"]})
    assert {x.receiver_mode for x in assign_trials(no_det, ["a"])} == {"watermark_only"}
