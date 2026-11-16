"""Training pipeline primitives used by ``scripts/train.py``.

This module implements the concrete steps of the AI learning system:

load dataset → validate → split → build pipeline → cross-validate →
fit → evaluate on held-out validation set → persist versioned artefact +
metrics. Nothing here is simulated: if no dataset exists the caller fails
loudly instead of pretending a model was trained.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from chatgangs.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class Dataset:
    """A labelled classification dataset."""

    name: str
    texts: List[str]
    labels: List[str]
    source: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.texts) != len(self.labels):
            raise ValueError(
                f"Dataset '{self.name}' has {len(self.texts)} texts but "
                f"{len(self.labels)} labels"
            )

    @property
    def size(self) -> int:
        return len(self.texts)

    def label_distribution(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for label in self.labels:
            counts[label] = counts.get(label, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))

    def validate(self, min_per_class: int = 3) -> List[str]:
        problems: List[str] = []
        if self.size == 0:
            problems.append("dataset is empty")
        distribution = self.label_distribution()
        for label, count in distribution.items():
            if count < min_per_class:
                problems.append(
                    f"class '{label}' only has {count} example(s); minimum is {min_per_class}"
                )
        if len(distribution) < 2:
            problems.append("at least two classes are required for classification")
        return problems

    def split(
        self, test_size: float = 0.2, seed: int = 42, stratify: bool = True
    ) -> Tuple["Dataset", "Dataset"]:
        """Deterministic train/validation split."""
        rng = np.random.default_rng(seed)
        train_idx: List[int] = []
        test_idx: List[int] = []
        if stratify:
            by_label: Dict[str, List[int]] = {}
            for idx, label in enumerate(self.labels):
                by_label.setdefault(label, []).append(idx)
            for label, indices in by_label.items():
                arr = np.array(indices)
                rng.shuffle(arr)
                if len(arr) < 2:
                    train_idx.extend(arr.tolist())
                    continue
                n_test = max(1, int(round(len(arr) * test_size)))
                test_idx.extend(arr[:n_test].tolist())
                train_idx.extend(arr[n_test:].tolist())
        else:
            arr = np.arange(self.size)
            rng.shuffle(arr)
            n_test = max(1, int(round(self.size * test_size)))
            test_idx = arr[:n_test].tolist()
            train_idx = arr[n_test:].tolist()
        rng.shuffle(train_idx)
        rng.shuffle(test_idx)
        return (
            Dataset(
                f"{self.name}:train",
                [self.texts[i] for i in train_idx],
                [self.labels[i] for i in train_idx],
            ),
            Dataset(
                f"{self.name}:validation",
                [self.texts[i] for i in test_idx],
                [self.labels[i] for i in test_idx],
            ),
        )


def _import_sklearn():
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
        )
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.pipeline import Pipeline

        return {
            "TfidfVectorizer": TfidfVectorizer,
            "LogisticRegression": LogisticRegression,
            "accuracy_score": accuracy_score,
            "classification_report": classification_report,
            "confusion_matrix": confusion_matrix,
            "f1_score": f1_score,
            "cross_val_score": cross_val_score,
            "StratifiedKFold": StratifiedKFold,
            "Pipeline": Pipeline,
        }
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "scikit-learn is required for training. Install it with:\n"
            "    pip install -r requirements.txt"
        ) from exc


class TextNormalizerTransformer:
    """scikit-learn transformer applying ChatGangs' real text normaliser.

    Using the same preprocessing at training time and inference time is what
    makes the classifier robust to slang, repeated characters and typos.
    """

    def __init__(self, language: str = "id") -> None:
        self.language = language
        self._normalizer = None

    def fit(self, X, y=None):  # noqa: N803 - sklearn API
        return self

    def transform(self, X):  # noqa: N803 - sklearn API
        if self._normalizer is None:
            from chatgangs.core.normalizer import get_normalizer

            self._normalizer = get_normalizer()
        return [self._normalizer.normalize(text, language=self.language) for text in X]

    def __getstate__(self):
        # The normaliser holds a thread lock, so it must not be pickled.
        state = self.__dict__.copy()
        state["_normalizer"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._normalizer = None

    def get_params(self, deep: bool = True):
        return {"language": self.language}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


def build_pipeline(seed: int = 42, max_features: int = 30000, normalize: bool = True):
    """Normaliser → TF-IDF (word + char n-grams) → Logistic Regression."""
    sk = _import_sklearn()
    # sklearn's FeatureUnion is used through a plain Pipeline with two feature
    # extractors via `ColumnTransformer`-free approach: word+char n-grams are
    # combined in a single vectoriser configuration by analysing both word and
    # character boundaries.
    word = sk["TfidfVectorizer"](
        analyzer="word",
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
        max_features=max_features,
        lowercase=True,
        strip_accents="unicode",
    )
    char = sk["TfidfVectorizer"](
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
        min_df=1,
        max_features=max_features,
        lowercase=True,
        strip_accents="unicode",
    )
    from sklearn.pipeline import FeatureUnion

    features = FeatureUnion([("word", word), ("char", char)])
    model = sk["LogisticRegression"](
        max_iter=3000,
        C=8.0,
        solver="lbfgs",
        class_weight="balanced",
        random_state=seed,
    )
    steps = []
    if normalize:
        steps.append(("normalizer", TextNormalizerTransformer()))
    steps.extend([("features", features), ("clf", model)])
    return sk["Pipeline"](steps)


@dataclass
class TrainingResult:
    task: str
    version: str
    trained_at: str
    dataset_size: int
    label_distribution: Dict[str, int]
    metrics: Dict[str, Any]
    model_path: Path
    report: str = ""
    duration_seconds: float = 0.0

    def metadata(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "version": self.version,
            "trained_at": self.trained_at,
            "dataset_size": self.dataset_size,
            "label_distribution": self.label_distribution,
            "metrics": self.metrics,
            "algorithm": "tfidf(word+char-ngram)+logistic-regression",
            "duration_seconds": round(self.duration_seconds, 2),
        }


def train_and_evaluate(
    dataset: Dataset,
    *,
    version: Optional[str] = None,
    test_size: float = 0.2,
    cv_folds: int = 5,
    seed: int = 42,
    models_dir: Optional[Path] = None,
) -> TrainingResult:
    """Train, cross-validate, evaluate and persist a classifier."""
    started = time.perf_counter()
    problems = dataset.validate(min_per_class=3)
    if problems:
        raise ValueError(
            "Dataset failed validation:\n  - " + "\n  - ".join(problems)
        )

    sk = _import_sklearn()
    train, validation = dataset.split(test_size=test_size, seed=seed)
    logger.info(
        "Training '%s': %d train / %d validation examples",
        dataset.name, train.size, validation.size,
    )

    pipeline = build_pipeline(seed=seed)

    # --- cross-validation on the training split (real, nested model fits)
    min_class = min(train.label_distribution().values())
    folds = max(2, min(cv_folds, min_class))
    cv_scores = sk["cross_val_score"](
        build_pipeline(seed=seed),
        train.texts,
        train.labels,
        cv=sk["StratifiedKFold"](n_splits=folds, shuffle=True, random_state=seed),
        scoring="f1_macro",
        n_jobs=1,
    )

    # --- final fit on the full training split
    pipeline.fit(train.texts, train.labels)

    # --- held-out evaluation
    predictions = pipeline.predict(validation.texts)
    accuracy = float(sk["accuracy_score"](validation.labels, predictions))
    macro_f1 = float(sk["f1_score"](validation.labels, predictions, average="macro", zero_division=0))
    weighted_f1 = float(sk["f1_score"](validation.labels, predictions, average="weighted", zero_division=0))
    report = sk["classification_report"](validation.labels, predictions, zero_division=0)
    labels = sorted(set(dataset.labels) | set(validation.labels))
    matrix = sk["confusion_matrix"](validation.labels, predictions, labels=labels).tolist()

    # --- training-set accuracy (sanity check for under/over-fitting)
    train_predictions = pipeline.predict(train.texts)
    train_accuracy = float(sk["accuracy_score"](train.labels, train_predictions))

    metrics = {
        "validation_accuracy": round(accuracy, 4),
        "validation_macro_f1": round(macro_f1, 4),
        "validation_weighted_f1": round(weighted_f1, 4),
        "train_accuracy": round(train_accuracy, 4),
        "cross_validation": {
            "folds": folds,
            "scoring": "f1_macro",
            "scores": [round(float(score), 4) for score in cv_scores],
            "mean": round(float(cv_scores.mean()), 4),
            "std": round(float(cv_scores.std()), 4),
        },
        "confusion_matrix": {"labels": labels, "matrix": matrix},
        "validation_size": validation.size,
        "train_size": train.size,
    }

    version = version or time.strftime("%Y%m%d.%H%M%S")
    trained_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    destination = Path(models_dir or Path(__file__).resolve().parents[2] / "models") / dataset.name.split(":")[0]
    destination.mkdir(parents=True, exist_ok=True)

    result = TrainingResult(
        task=dataset.name,
        version=version,
        trained_at=trained_at,
        dataset_size=dataset.size,
        label_distribution=dataset.label_distribution(),
        metrics=metrics,
        model_path=destination / "model.joblib",
        report=report,
        duration_seconds=time.perf_counter() - started,
    )

    import joblib

    bundle = {
        "model": pipeline.named_steps["clf"],
        "vectorizer": pipeline.named_steps["features"],
        "pipeline": pipeline,
        "metadata": result.metadata(),
    }
    joblib.dump(bundle, result.model_path)
    (destination / "metrics.json").write_text(
        json.dumps(result.metadata(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (destination / "classification_report.txt").write_text(report, encoding="utf-8")
    return result


# --------------------------------------------------------------------------- #
# Dataset loaders
# --------------------------------------------------------------------------- #
def load_intent_dataset(path: Path) -> Dataset:
    """Build the intent dataset from ``data/intents/intents.json``."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    texts: List[str] = []
    labels: List[str] = []
    for intent in payload.get("intents", []):
        for example in intent.get("examples", []):
            if example.strip():
                texts.append(example.strip())
                labels.append(intent["name"])
    return Dataset("intent", texts, labels)


# Labels accepted by the sentiment task. Other tasks are free-form topics.
SENTIMENT_LABELS = frozenset({"positive", "negative", "neutral", "positif", "negatif", "netral"})


def load_csv_dataset(path: Path, name: str, allowed_labels=None) -> Dataset:
    """Load a ``text,label`` CSV file.

    ``allowed_labels`` restricts the accepted label vocabulary (used by the
    sentiment task); when omitted any non-empty label is kept, which is what
    free-form topic classification needs.
    """
    import csv

    texts: List[str] = []
    labels: List[str] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"text", "label"}
        if not required.issubset({c.lower() for c in (reader.fieldnames or [])}):
            raise ValueError(f"{path} must contain columns: text,label")
        for row in reader:
            normalized = {(k or "").lower(): v for k, v in row.items()}
            text = (normalized.get("text") or "").strip()
            label = (normalized.get("label") or "").strip().lower()
            if not text or not label:
                continue
            if allowed_labels is not None and label not in allowed_labels:
                continue
            texts.append(text)
            labels.append(label)
    return Dataset(name, texts, labels)


def load_sentiment_dataset(path: Path) -> Dataset:
    """Load the sentiment dataset (positive/negative/neutral only)."""
    return load_csv_dataset(path, "sentiment", SENTIMENT_LABELS)


def load_classification_dataset(path: Path) -> Optional[Dataset]:
    """Load the topic-classification dataset (free-form labels)."""
    if not path.exists():
        return None
    return load_csv_dataset(path, "classification")


def evaluate_intent_detector(verbose: bool = True) -> Dict[str, Any]:
    """Evaluate the *live* detector (rules + optional model) on the catalogue.

    This measures end-to-end behaviour rather than only the classifier, which
    is what users actually experience.
    """
    from chatgangs.config import get_settings
    from chatgangs.core.intent import get_intent_detector

    settings = get_settings()
    dataset = load_intent_dataset(Path(settings.data_dir) / "intents" / "intents.json")
    detector = get_intent_detector()
    correct = 0
    per_class: Dict[str, List[int]] = {}
    failures: List[Tuple[str, str, str]] = []
    for text, expected in zip(dataset.texts, dataset.labels):
        predicted = detector.detect(text, "id").name
        per_class.setdefault(expected, [0, 0])
        per_class[expected][1] += 1
        if predicted == expected:
            correct += 1
            per_class[expected][0] += 1
        else:
            failures.append((text, expected, predicted))
    accuracy = correct / max(dataset.size, 1)
    result = {
        "dataset_size": dataset.size,
        "accuracy": round(accuracy, 4),
        "per_intent_accuracy": {
            label: round(hits / total, 4) for label, (hits, total) in sorted(per_class.items())
        },
        "failures_sample": failures[:10],
        "model_available": detector.model_available,
    }
    if verbose:
        print(f"End-to-end intent accuracy: {accuracy:.2%} ({correct}/{dataset.size})")
        for label, (hits, total) in sorted(per_class.items()):
            if hits < total:
                print(f"  {label:<18} {hits}/{total}")
    return result
