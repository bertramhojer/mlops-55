# Project Description

**Goal:**
The primary purpose of this project is to finetune a cutting-edge natural language processing model to perform binary or multiclass classification on a multiple-choice Q&A dataset for knowledge evaluation across various subjects. The secondary purpose is for the authors of this repository to further develop their MLOps skills by employing a variety of tools for model development, deployment, and maintenance.

**Framework:**
We use HuggingFace’s transformers library to load the pre-trained ModernBERT embeddings and PyTorch-Lightning to fine-tune the model on the specified dataset. Weights and Biases will be utilized to optimize ModernBERT’s hyperparameters, and resulting models will be evaluated with several Scikit-learn metrics. For packaging management and development, we employ uv and Docker dev containers to ensure every group member starts with the same environment configuration when working on the project.

**Data:**
We use HuggingFace datasets to load the MMLU (Massive Multitask Language Understanding)  dataset. The dataset is preprocessed to fine-tune a BERT model. We provide both a binary and multi-class format of the data. The binary format dataset is 4x larger, providing certain benefits: more training data and a more explicit learning signal from negative and positive examples. The multi-class format dataset only contains correct question-answer pairs, resulting in faster inference than the binary format, as we only need one inference to pass for classification (instead of four).

**Deep learning models used:**
ModernBERT is a state-of-the-art bidirectional encoder-only Transformer pre-trained on 2 trillion tokens of English and code data. ModernBERT supports a context length of 8,192 tokens, offering superior downstream performance and faster processing than older encoder models. In addition, we will integrate different classification heads. Released only a few weeks ago, ModernBERT has yet to be tested on MMLU benchmarks. For this project, we incorporate different classification heads to achieve satisfactory classification accuracy, aligning with our focus on MLOps rather than model complexity.

**References:**
- [HuggingFace](https://huggingface.co/answerdotai/ModernBERT-base)
- [MMLU Dataset](https://huggingface.co/datasets/cais/mmlu)
- [Lightning AI](https://lightning.ai/)
- [Package Manager - UV](https://docs.astral.sh/uv/)
- [Dev Container Docs](https://code.visualstudio.com/docs/devcontainers/containers)

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
