"""wm-eval command line. Every command gets a run_id and writes JSONL events."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from wm_eval.config import load_config, load_dotenv
from wm_eval.observability.events import Recorder, setup_logging
from wm_eval.observability.storage import environment_info, new_run_id

DEFAULT_CONFIGS = Path("configs")


def _command_recorder(root: str, name: str) -> Recorder:
    run_id = new_run_id(name)
    return Recorder(Path(root) / "_commands" / f"{run_id}.jsonl", run_id)


def cmd_doctor(args: argparse.Namespace) -> int:
    import yaml

    from wm_eval.config import ReceiverSpec, Settings
    from wm_eval.receiver.llama_client import LlamaClient

    settings = Settings(**(yaml.safe_load(open(args.configs / "default.yaml")) or {}))
    rec = _command_recorder(settings.output_root, "doctor")
    checks: list[tuple[str, bool, str]] = []
    env = environment_info()
    checks.append(("python >= 3.11", sys.version_info >= (3, 11), sys.version.split()[0]))
    for pkg, ver in env["packages"].items():
        checks.append((f"package {pkg}", ver is not None, ver or "missing"))
    gpus = env.get("gpus") or []
    checks.append(
        (
            "compute device",
            True,
            ", ".join(f"{g['name']} ({g['memory_gb']} GB)" for g in gpus) or "CPU only",
        )
    )
    models = yaml.safe_load(open(args.configs / "models.yaml")) or {}
    for name, spec in (models.get("receivers") or {}).items():
        h = LlamaClient(ReceiverSpec(**spec)).health()
        detail = (
            f"{spec.get('base_url')} model_path={h.get('model_path')}"
            if h.get("ok")
            else h.get("error", "")
        )
        checks.append((f"receiver {name}", bool(h.get("ok")), detail))
    for name, spec in (models.get("generators") or {}).items():
        cached = _hf_cached(spec["hf_id"], spec.get("revision", "main"))
        checks.append(
            (
                f"generator {name} weights cached",
                cached,
                f"{spec['hf_id']}@{spec.get('revision', 'main')[:10]}",
            )
        )
    wm = yaml.safe_load(open(args.configs / "watermark.yaml")) or {}
    for wid, w in wm.items():
        env_name = w.get("secret_env", "WM_SECRET")
        checks.append(
            (
                f"watermark {wid} secret ({env_name})",
                bool(os.environ.get(env_name)),
                "set" if os.environ.get(env_name) else "not set",
            )
        )
    out = Path(settings.output_root)
    try:
        out.mkdir(parents=True, exist_ok=True)
        probe = out / ".doctor_probe"
        probe.write_text("ok")
        probe.unlink()
        checks.append(("output dir writable", True, str(out.resolve())))
    except OSError as e:
        checks.append(("output dir writable", False, repr(e)))
    for name, ok, detail in checks:
        print(f"{'OK ' if ok else 'XX '} {name:40s} {detail}")
    rec.emit("doctor.completed", data={"checks": [{"name": n, "ok": o, "detail": d} for n, o, d in checks],
                                      "environment": env})  # fmt: skip
    print(f"\nrun_id={rec.run_id} events={rec.path}")
    # Receivers/generators may legitimately be offline; only hard failures fail doctor.
    hard = [n for n, o, _ in checks if not o and (n.startswith(("python", "package", "output")))]
    return 1 if hard else 0


def _hf_cached(repo_id: str, revision: str) -> bool:
    """True only if the pinned revision's snapshot holds complete safetensors weights."""
    try:
        from huggingface_hub import scan_cache_dir

        for repo in scan_cache_dir().repos:
            if repo.repo_id == repo_id:
                for rev in repo.revisions:
                    if (rev.commit_hash == revision or revision in rev.refs) and any(
                        f.file_name.endswith(".safetensors") for f in rev.files
                    ):
                        return True
    except Exception:
        pass
    return False


def cmd_run(args: argparse.Namespace) -> int:
    from wm_eval.experiments.runner import STAGES, run

    stages = tuple(args.stages.split(",")) if args.stages else STAGES
    bad = set(stages) - set(STAGES)
    if bad:
        print(f"unknown stages {bad}; choose from {STAGES}", file=sys.stderr)
        return 2
    if args.resume:
        rd = run(run_id=args.resume, root=args.output_root or "results", stages=stages)
    else:
        if not args.config:
            print("--config is required for a new run", file=sys.stderr)
            return 2
        cfg = load_config(args.config, args.configs if args.configs_given else None)
        rd = run(cfg, root=args.output_root, stages=stages)
    print(f"run_id={rd.run_id}\nrun_dir={rd.path}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from wm_eval.analysis.report import build_report

    run_path = Path(args.output_root or "results") / args.run_id
    rec = Recorder(run_path / "events.jsonl", args.run_id)
    rec.emit("report.started")
    path = build_report(run_path)
    rec.emit("report.completed", data={"report": str(path)})
    print(path)
    return 0


def cmd_detect(args: argparse.Namespace) -> int:
    """Detect on a text file, retokenized with the generator's tokenizer."""
    import yaml

    from wm_eval.config import GeneratorSpec, WatermarkConfig
    from wm_eval.generation.models import load_tokenizer
    from wm_eval.watermark.detector import detect_text

    models = yaml.safe_load(open(args.configs / "models.yaml"))
    spec = GeneratorSpec(**models["generators"][args.generator])
    wm = WatermarkConfig(
        config_id=args.watermark,
        **yaml.safe_load(open(args.configs / "watermark.yaml"))[args.watermark],
    )
    rec = _command_recorder("results", "detect")
    res = detect_text(Path(args.text_file).name, Path(args.text_file).read_text(), load_tokenizer(spec), wm,
                      spec.tokenizer_id or spec.hf_id)  # fmt: skip
    rec.emit("watermark.detection.completed", data=res.model_dump())
    print(json.dumps(res.model_dump(), indent=2))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Score classifier evaluators against the hand-labelled set."""
    from wm_eval.evaluation.validation import validate

    rec = _command_recorder("results", "validate")
    results = validate(args.annotations)
    for r in results:
        print(
            f"{r['kind']:14s} n={r['n']:3d} accuracy={r['accuracy']:.2f} macro_f1={r['macro_f1']:.2f}  {r['evaluator_version']}"
        )
        for e in r["errors"]:
            print(f"    miss: {e}")
    rec.emit("evaluators.validated", data={"annotations": args.annotations, "results": results})
    print(f"\nrun_id={rec.run_id} events={rec.path}")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    """PDF / HTML / text files -> a JSONL of SourceDocuments for `experiment.sources`."""
    from wm_eval.data.ingest import ingest_files, save_jsonl

    files = [f for p in args.paths for f in (sorted(Path(p).glob("*")) if Path(p).is_dir() else [Path(p)])
             if f.suffix.lower() in {".pdf", ".html", ".htm", ".txt", ".md"}]  # fmt: skip
    if not files:
        print("no .pdf, .html, .txt or .md files found")
        return 1
    docs = ingest_files(files, args.domain, args.max_words, args.licence, args.url)
    save_jsonl(docs, args.out)
    for d in docs:
        p = d.provenance
        note = f" (trimmed from {p['words_extracted']})" if p["trimmed_to_words"] else ""
        hidden = (
            f", {p['invisible_chars_removed']} invisible chars removed"
            if p["invisible_chars_removed"] != "0"
            else ""
        )
        print(f"{d.source_id:40s} {len(d.text.split()):5d} words{note}{hidden}")
    print(f"\nwrote {len(docs)} documents to {args.out}. Check the text before running; set "
          f"experiment.sources: {args.out}")  # fmt: skip
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wm-eval", description=__doc__)
    p.add_argument("--configs", type=Path, help="configs directory (default: ./configs)")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="check environment, devices, model servers, output dirs")
    r = sub.add_parser("run", help="run (or resume) an experiment")
    r.add_argument("--config", help="experiment YAML (new run)")
    r.add_argument(
        "--resume", metavar="RUN_ID", help="resume an existing run with its saved config"
    )
    r.add_argument(
        "--stages", help="comma list of generate,detect,receive,evaluate,assemble (default: all)"
    )
    r.add_argument("--output-root", help="override settings.output_root")
    rp = sub.add_parser("report", help="build figures, tables and report.md for a run")
    rp.add_argument("--run-id", required=True)
    rp.add_argument("--output-root")
    d = sub.add_parser("detect", help="run SynthID detection on a text file")
    d.add_argument("--text-file", required=True)
    d.add_argument(
        "--generator", required=True, help="generator name from models.yaml (its tokenizer is used)"
    )
    d.add_argument("--watermark", default="research_default", help="config id from watermark.yaml")
    i = sub.add_parser("ingest", help="turn PDF/HTML/text files into a sources JSONL")
    i.add_argument("paths", nargs="+", help="files or directories (.pdf, .html, .txt, .md)")
    i.add_argument("--out", required=True, help="new JSONL file to write (never overwritten)")
    i.add_argument(
        "--domain", default="general", help="e.g. news, environment, politics, neuroscience"
    )
    i.add_argument("--max-words", type=int, default=2000,
                   help="trim each document (default 2000; ~8k prompt tokens is the 6 GB limit)")  # fmt: skip
    i.add_argument(
        "--licence", default="", help="licence recorded with every document, e.g. CC-BY-4.0"
    )
    i.add_argument("--url", default="", help="source URL recorded with every document")
    v = sub.add_parser(
        "validate-evaluators", help="score classifier evaluators against annotated examples"
    )
    v.add_argument("--annotations", default="data/annotations/evaluator_gold.jsonl")
    return p


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    # `run` finds configs next to the experiment file unless --configs was given explicitly.
    args.configs_given = args.configs is not None
    args.configs = args.configs or DEFAULT_CONFIGS
    setup_logging(os.environ.get("WM_EVAL_LOG_LEVEL", "INFO"))
    return {
        "doctor": cmd_doctor,
        "run": cmd_run,
        "ingest": cmd_ingest,
        "report": cmd_report,
        "detect": cmd_detect,
        "validate-evaluators": cmd_validate,
    }[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
