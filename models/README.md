# Trained model artefacts

`python -m scripts.train --task all` writes one folder per task:

```
models/
├── intent/
│   ├── model.joblib                 # TF-IDF(word+char) → LogisticRegression pipeline
│   ├── metrics.json                 # accuracy, macro/weighted F1, CV folds, confusion matrix
│   └── classification_report.txt
├── sentiment/
│   ├── model.joblib
│   ├── metrics.json
│   └── classification_report.txt
└── classification/
    ├── model.joblib
    ├── metrics.json
    └── classification_report.txt
```

The API loads these artefacts at start-up and reports their version, training
timestamp and validation accuracy at `GET /api/model/info` (field
`artifacts`). If a folder is missing, the corresponding module falls back to
its rule/lexicon engine and says so explicitly — no capability is faked.

Model binaries are ignored by git (see `.gitignore`) because they are
reproducible from the versioned datasets in `data/`.
