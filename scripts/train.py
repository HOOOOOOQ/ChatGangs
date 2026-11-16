#!/usr/bin/env python
"""ChatGangs model training pipeline (real training, no simulation).

Usage
-----
    python -m scripts.train --task intent
    python -m scripts.train --task sentiment
    python -m scripts.train --task classification
    python -m scripts.train --task all

What a run actually does
------------------------
1. Loads the dataset from ``data/`` and validates it (minimum examples per
   class, at least two classes).
2. Splits it deterministically into train / validation sets.
3. Runs stratified k-fold cross-validation (macro-F1) on the training split.
4. Fits TF-IDF (word + character n-grams) → Logistic Regression.
5. Evaluates accuracy / macro-F1 / weighted-F1 / per-class report and a
   confusion matrix on the *held-out* validation set.
6. Writes ``models/<task>/model.joblib``, ``metrics.json`` and
   ``classification_report.txt`` — the artefacts the API reads at runtime.

Exit codes: 0 on success, 1 on validation failure, 2 on missing dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chatgangs.config import get_settings  # noqa: E402
from chatgangs.logging_config import get_logger  # noqa: E402
from chatgangs.training.pipeline import (  # noqa: E402
    Dataset,
    evaluate_intent_detector,
    load_classification_dataset,
    load_intent_dataset,
    load_sentiment_dataset,
    train_and_evaluate,
)

logger = get_logger("chatgangs.training")


def _dataset_paths() -> Dict[str, Path]:
    data_dir = get_settings().data_dir
    return {
        "intent": data_dir / "intents" / "intents.json",
        "sentiment": data_dir / "sentiment" / "train.csv",
        "classification": data_dir / "classification" / "train.csv",
    }


def _print_result(result) -> None:
    metrics = result.metrics
    print("\n" + "=" * 74)
    print(f"  Model trained : {result.task}")
    print(f"  Version       : {result.version}")
    print(f"  Dataset size  : {result.dataset_size}")
    print(f"  Labels        : {', '.join(result.label_distribution)}")
    print(f"  Artefact      : {result.model_path}")
    print("-" * 74)
    print(f"  Validation acc: {metrics['validation_accuracy']:.2%}")
    print(f"  Macro F1      : {metrics['validation_macro_f1']:.4f}")
    print(f"  Weighted F1   : {metrics['validation_weighted_f1']:.4f}")
    print(f"  Train acc     : {metrics['train_accuracy']:.2%}")
    cv = metrics["cross_validation"]
    print(
        f"  CV ({cv['folds']}-fold) : mean {cv['mean']:.4f} ± {cv['std']:.4f} "
        f"[{', '.join(f'{s:.3f}' for s in cv['scores'])}]"
    )
    print(f"  Duration      : {result.duration_seconds:.1f}s")
    print("=" * 74)
    if result.report:
        print("\nPer-class report (held-out validation set):\n")
        print(result.report)


def run_task(task: str, version: Optional[str], test_size: float, cv_folds: int) -> int:
    paths = _dataset_paths()
    path = paths[task]
    if not path.exists():
        print(f"✗ Dataset not found: {path}", file=sys.stderr)
        return 2

    if task == "intent":
        dataset = load_intent_dataset(path)
    elif task == "classification":
        dataset = load_classification_dataset(path)
    else:
        dataset = load_sentiment_dataset(path)

    try:
        result = train_and_evaluate(
            dataset, version=version, test_size=test_size, cv_folds=cv_folds
        )
    except ValueError as exc:
        print(f"✗ {exc}", file=sys.stderr)
        return 1
    _print_result(result)
    return 0


def evaluate_qa() -> int:
    """Evaluate the retrieval-based QA module against the labelled set."""
    from chatgangs.core.semantics import get_knowledge_base

    path = get_settings().data_dir / "qa" / "evaluation_questions.jsonl"
    if not path.exists():
        print(f"✗ QA evaluation set not found: {path}", file=sys.stderr)
        return 2
    kb = get_knowledge_base()
    total = correct = 0
    top3 = 0
    failures = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        matches = kb.retrieve(item["question"], language="id", top_k=3)
        predicted = matches[0].entry["id"] if matches else None
        total += 1
        if predicted == item["expected"]:
            correct += 1
        if any(m.entry["id"] == item["expected"] for m in matches):
            top3 += 1
        else:
            failures.append((item["question"], item["expected"], predicted))
    print("\n" + "=" * 74)
    print("  Question Answering evaluation")
    print(f"  Questions      : {total}")
    print(f"  Top-1 accuracy : {correct / max(total, 1):.2%} ({correct}/{total})")
    print(f"  Top-3 recall   : {top3 / max(total, 1):.2%} ({top3}/{total})")
    print("=" * 74)
    if failures:
        print("\nMissed questions:")
        for question, expected, predicted in failures:
            print(f"  ✗ {question!r} expected={expected} predicted={predicted}")
    return 0


def validate_only() -> int:
    """Load and validate every dataset without training anything."""
    paths = _dataset_paths()
    exit_code = 0
    checks = [
        ("intent", paths["intent"], load_intent_dataset(paths["intent"])),
        ("sentiment", paths["sentiment"], load_sentiment_dataset(paths["sentiment"])),
        (
            "classification",
            paths["classification"],
            load_classification_dataset(paths["classification"]),
        ),
    ]
    for task, path, dataset in checks:
        problems = dataset.validate()
        status = "ok" if not problems else "INVALID: " + "; ".join(problems)
        print(f"{task:>15}  {len(dataset.texts):>5} examples  {status}  ({path})")
        if problems:
            exit_code = 1
    print("\nDry run: no model was trained and no artefact was written.")
    return exit_code


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="ChatGangs training pipeline")
    parser.add_argument(
        "--task",
        choices=["intent", "sentiment", "classification", "qa", "all", "evaluate"],
        default="all",
        help="which model to train (default: all)",
    )
    parser.add_argument("--version", default=None, help="model version label")
    parser.add_argument("--test-size", type=float, default=0.2, help="validation split ratio")
    parser.add_argument("--cv-folds", type=int, default=5, help="cross-validation folds")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate datasets only; do not fit or overwrite any artefact",
    )
    args = parser.parse_args(argv)

    started = time.perf_counter()
    exit_code = 0

    if args.dry_run:
        return validate_only()

    if args.task in ("intent", "all"):
        exit_code |= run_task("intent", args.version, args.test_size, args.cv_folds)
    if args.task in ("sentiment", "all"):
        exit_code |= run_task("sentiment", args.version, args.test_size, args.cv_folds)
    if args.task in ("classification", "all"):
        exit_code |= run_task("classification", args.version, args.test_size, args.cv_folds)
    if args.task in ("qa", "all", "evaluate"):
        exit_code |= evaluate_qa()
    if args.task in ("intent", "all", "evaluate"):
        print()
        evaluate_intent_detector(verbose=True)

    print(f"\nFinished in {time.perf_counter() - started:.1f}s")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
