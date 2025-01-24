<div align="center">

<br/>

<p align="center">
<a href="https://pytorch.org/get-started/locally/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.12-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 2.5+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?style=for-the-badge&logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: hydra" src="https://img.shields.io/badge/config-hydra-89b8cd?style=for-the-badge&labelColor=gray"></a>
<a href="https://beta.ruff.rs/docs/"><img alt="Code style: Ruff" src="https://img.shields.io/badge/code%20style-Ruff-007ACC.svg?style=for-the-badge&labelColor=gray"></a>
</div>

<br/>

<div align="center">
MLOps-DTU: <a href="https://skaftenicki.github.io/dtu_mlops/">Machine Learning Operations course 02476 at DTU.</a>
</div>

<br/>

**👉 Click the image to visit the Frontend Application!**

[![Frontend Application](https://github.com/bertramhojer/mlops-55/blob/main/reports/figures/streamlit.png)](https://modernbert-frontend-203438086142.europe-west1.run.app)

# Introduction
<details>
<summary>🐙 Project Description</summary>

**Goal:**
The primary purpose of this project is to finetune a cutting-edge natural language processing model to perform binary or multiclass classification on a multiple-choice Q&A dataset for knowledge evaluation across various subjects. The secondary purpose is for the authors of this repository to further develop their MLOps skills by employing various tools for model development, deployment, and maintenance.

**Framework:**
We use HuggingFace’s transformers library to load the pre-trained [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base) embeddings and [PyTorch-Lightning](https://lightning.ai/) to fine-tune the model on the specified dataset. Weights and Biases will be utilized to optimize ModernBERT’s hyperparameters, and resulting models will be evaluated with several Scikit-learn metrics. For packaging management and development, we employ [uv](https://docs.astral.sh/uv/) and [Docker Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers) to ensure every group member starts with the same environment configuration when working on the project.

**Data:**
We use HuggingFace datasets to load the MMLU (Massive Multitask Language Understanding)  dataset. The dataset is preprocessed to fine-tune a BERT model. We provide both a binary and multi-class format of the data. The binary format dataset is 4x larger, providing certain benefits: more training data and a more explicit learning signal from negative and positive examples. The multi-class format dataset only contains correct question-answer pairs, resulting in faster inference than the binary format, as we only need one inference to pass for classification (instead of four).

**Deep learning models used:**
ModernBERT is a state-of-the-art bidirectional encoder-only Transformer pre-trained on 2 trillion tokens of English and code data. ModernBERT supports a context length of 8,192 tokens, offering superior downstream performance and faster processing than older encoder models. In addition, we will integrate different classification heads. Released only a few weeks ago, ModernBERT has yet to be tested on MMLU benchmarks. For this project, we incorporate different classification heads to achieve satisfactory classification accuracy, aligning with our focus on MLOps rather than model complexity.

</div>
</details>

<details>
<summary>📦 Project structure</summary>

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
│   ├── frontend.Dockerfile
│   ├── preprocess.Dockerfile
│   └── train.Dockerfile
├── models/                   # Trained models
├── src/                      # Source code
│   └── project
│       ├── __init__.py
│       ├── __pycache__
│       ├── api.py
│       ├── collate.py
│       ├── configs.py
│       ├── data.py
│       ├── data_drift.py
│       ├── evaluate.py
│       ├── frontend.py
│       ├── model.py
│       ├── settings.py
│       ├── sweep.py
│       ├── tools.py
│       ├── train.py
│       ├── utils.py
│       └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_model.py
│   └── test_train.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── ruff.toml            # Ruff settings for linting
├── uv.lock            # Checkpoint of python dependencies
├── README.md                 # Project README
```
</details>

<details>
<summary>📚 Report</summary>

**👉 [Click to see our report and get more insight on the project!](https://github.com/bertramhojer/mlops-55/tree/main/reports)**


</details>


# Setup 🔧

<details>
<summary>Installation</summary>

We only support development mode for now. You need to install install `uv` and the project using:
```shell
curl -LsSf https://astral.sh/uv/install.sh | less
uv sync
```
As a fast track, we have defined a dev container, which will spin a containarized environment ready for development.

If you only want to use the project for trainig or frontend, you can pass in the `--extra` argument. For examples, to use the project for training:
```shell
uv sync --extra train
```

</details>

# Usage 🦊
<details>
<summary>Train</summary>

The `train` endpoint uses `hydra` to parse arguments and configure the run.
See `src/project/train.py` for the default configuration. You can override any of the default values by passing them as arguments to the `train` endpoint. For example, to train a T5-base encoder on MMLU using DDP:


```shell
uv run train train=distributed_train train.model_name=t5-base train.batch_size=4 train.strategy=DDP datamodule.file_name=mmlu 
```
You can also define a yaml configuration named "expA.yaml" in the folder `configs/experiments`, and use it to override the default variables:
```shell
uv run train experiment=expA
```

</details>



Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
