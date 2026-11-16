"""Training pipeline tests — real training on small, deterministic datasets."""

from pathlib import Path

import pytest

from chatgangs.training.pipeline import (
    Dataset,
    build_pipeline,
    load_classification_dataset,
    load_intent_dataset,
    load_sentiment_dataset,
    train_and_evaluate,
)


def toy_dataset() -> Dataset:
    texts, labels = [], []
    samples = {
        "greeting": ["halo", "hai", "selamat pagi", "hello there", "apa kabar"],
        "farewell": ["sampai jumpa", "bye", "selamat tinggal", "goodbye", "see you"],
        "pricing": ["berapa harga", "biaya berapa", "is it free", "harga paket", "pricing"],
    }
    for label, examples in samples.items():
        for example in examples:
            texts.append(example)
            labels.append(label)
    return Dataset("toy", texts, labels)


def test_classification_loader_accepts_free_form_labels(tmp_path):
    """Regression: the topic dataset must not be filtered by sentiment labels."""
    path = tmp_path / "topics.csv"
    path.write_text(
        "text,label\n"
        "chip baru lebih cepat,teknologi\n"
        "harga cabai naik,bisnis\n"
        "film horor laris,hiburan\n",
        encoding="utf-8",
    )
    dataset = load_classification_dataset(path)
    assert dataset is not None
    assert dataset.size == 3
    assert set(dataset.labels) == {"teknologi", "bisnis", "hiburan"}

    # the sentiment loader keeps its strict label filter
    sentiment = load_sentiment_dataset(path)
    assert sentiment.size == 0


def test_dataset_validation():
    assert toy_dataset().validate() == []
    with pytest.raises(ValueError):
        Dataset("broken", ["a", "b"], ["one"])
    assert Dataset("tiny", ["a"], ["one"]).validate()


def test_stratified_split_is_deterministic():
    dataset = toy_dataset()
    a_train, a_val = dataset.split(seed=1)
    b_train, b_val = dataset.split(seed=1)
    assert a_train.texts == b_train.texts
    assert a_val.texts == b_val.texts
    assert a_train.size + a_val.size == dataset.size


def test_training_produces_real_metrics(tmp_path):
    result = train_and_evaluate(toy_dataset(), models_dir=tmp_path, cv_folds=3)
    assert result.model_path.exists()
    assert (tmp_path / "toy" / "metrics.json").exists()
    metrics = result.metrics
    assert 0.0 <= metrics["validation_accuracy"] <= 1.0
    assert metrics["cross_validation"]["folds"] >= 2
    assert len(metrics["confusion_matrix"]["matrix"]) == 3
    assert result.label_distribution["greeting"] == 5


def test_training_rejects_invalid_dataset(tmp_path):
    with pytest.raises(ValueError):
        train_and_evaluate(Dataset("bad", ["a", "b"], ["x", "x"]), models_dir=tmp_path)


def test_trained_pipeline_predicts(tmp_path):
    result = train_and_evaluate(toy_dataset(), models_dir=tmp_path, cv_folds=3)
    import joblib

    bundle = joblib.load(result.model_path)
    pipeline = bundle["pipeline"]
    assert pipeline.predict(["hello"])[0] == "greeting"
    assert pipeline.predict(["berapa biayanya"])[0] == "pricing"


def test_intent_dataset_loader():
    settings_path = Path(__file__).resolve().parents[1] / "data" / "intents" / "intents.json"
    dataset = load_intent_dataset(settings_path)
    assert dataset.size > 300
    assert dataset.validate() == []
    assert len(dataset.label_distribution()) >= 10


def test_sentiment_dataset_loader():
    path = Path(__file__).resolve().parents[1] / "data" / "sentiment" / "train.csv"
    dataset = load_sentiment_dataset(path)
    assert dataset.size > 100
    assert set(dataset.label_distribution()) <= {"positive", "negative", "neutral"}


def test_build_pipeline_has_normaliser_step():
    pipeline = build_pipeline(normalize=True)
    assert "normalizer" in pipeline.named_steps
    assert "features" in pipeline.named_steps
    assert "clf" in pipeline.named_steps
