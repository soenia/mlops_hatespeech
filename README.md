# mlops_hatespeech
Text classification of hate speech for MLOps course summer term 2025.

## Project links

You can access the application here:
[Go to the App](https://frontend-178847025464.europe-west1.run.app/)

Project documentation is available here:
[View the Documentation](https://soenia.github.io/mlops_hatespeech/)

## Project description

The goal of the project is to use Machine Learning methods to identify hatespeech in given input strings.
We leverage transformer based models for the binary classification task.
We use a small version of BERT able to run on CPU as a finetuning basis.
We are using a labeled [Huggingface dataset](https://huggingface.co/datasets/thefrankhsu/hate_speech_twitter) containing tweets. The dataset consisits of a training set with 5679 tweets and a test set of 1000 tweets.

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                        # Github actions
│   └── workflows/
│       ├── cml_data.yaml
│       ├── codecheck.yaml
│       ├── deploy_docs.yaml
│       └── tests.yaml
├── cloud/                          # Cloud Configuration files
│   ├── cloudbuild_app.yaml
│   ├── cloudbuild_data.yaml
│   ├── cloudbuild_evaluate.yaml
│   ├── cloudbuild_train.yaml
│   ├── vertex_config_data.yaml
│   ├── vertex_config_evaluate.yaml
│   ├── vertex_config_train.yaml
│   └── vertex_train_start.yaml
├── configs/                        # Configuration files
├── data/                           # Data directory
│   └── processed/                  # Data splits
│       ├── test/
│       ├── train/
│       └── validation/
├── dockerfiles/                    # Dockerfiles
│   ├── app.Dockerfile
│   ├── bento.Dockerfile
│   ├── data.Dockerfile
│   ├── evaluate.Dockerfile
│   ├── frontend.Dockerfile
│   └── train.Dockerfile
├── docs/                           # Documentation
├── logs/                           # Model logs (Evaluation & Checkpoints)
│   ├── eval/
│   └── run1/
├── reports/                        # Reports
│   ├── figures/
│   └── report.html
├── src/                            # Source code
│   └── mlops_hatespeech/
│       ├── __init__.py
│       ├── app.py                  # App via FastAPI
│       ├── bentoml_service.py      # App via BentoML
│       ├── create_onnx.py          # Create .onnx from our model
│       ├── data.py                 # Preprocessing
│       ├── dataset_statistics.py   # Creates dataset statistics report
│       ├── drift_detector.py       # Creates data drift report
│       ├── evaluate.py             # Takes test split and evaluates performance
│       ├── frontend.py             # Frontend of our app
│       ├── logger.py               # Logger for the project
│       ├── model.py                # model string of our model
│       └── train.py                # Takes train split and trains the model
├── tests/                          # Tests
│   ├── integrationtests/
│   │   └──  test_api.py            # Basic API test
│   ├── performancetests/
│   │   └──  locustfile.py          # Load test
│   ├── __init__.py
│   ├── test_data.py
│   └── test_model.py
├── wandb/                          # Weights & Biases logs
├── .cloudignore
├── .dockerignore
├── .dvcignore
├── .gitignore
├── .pre-commit-config.yaml
├── config.yaml
├── LICENSE
├── mkdocs.yaml                     # Configuration for the documentation
├── pyproject.toml                  # Python project file
├── README.md                       # Project README
├── requirements.txt                # Project requirements
├── requirements_dev.txt            # Development requirements
├── requirements_test.txt           # Test requirements
├── requirements_frontend.txt       # Frontend requirements
└── tasks.py                        # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
