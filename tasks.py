import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "mlops_hatespeech"
PYTHON_VERSION = "3.11"


# Setup commands
@task
def create_environment(ctx: Context) -> None:
    """Create a new conda environment for project."""
    ctx.run(
        f"conda create --name {PROJECT_NAME} python={PYTHON_VERSION} pip --no-default-packages --yes",
        echo=True,
        pty=not WINDOWS,
    )


@task
def requirements(ctx: Context) -> None:
    """Install project requirements."""
    ctx.run("pip install -U pip setuptools wheel", echo=True, pty=not WINDOWS)
    ctx.run("pip install -e .", echo=True, pty=not WINDOWS)


@task(requirements)
def dev_requirements(ctx: Context) -> None:
    """Install development requirements."""
    ctx.run('pip install -e .["dev"]', echo=True, pty=not WINDOWS)


@task
def test_requirements(ctx: Context) -> None:
    """Install test dependencies."""
    ctx.run("pip install -r requirements_tests.txt", echo=True, pty=not WINDOWS)


# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(f"python src/{PROJECT_NAME}/data.py", echo=True, pty=not WINDOWS)


@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"python src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("coverage report -m -i", echo=True, pty=not WINDOWS)


@task
def get_model(ctx: Context) -> None:
    """Downloads best of our checkpoints and generates onnx."""
    ctx.run(f"python src/{PROJECT_NAME}/create_onnx.py", echo=True, pty=not WINDOWS)


# We only show one example on how this would work.
# bento-app is the most independent version with few dependencies:
# Note that you still need gcloud credentials for this to work.
@task
def docker_build_bento(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        (
            "docker buildx build "
            "--platform=linux/amd64 "
            "-t bento-app "
            "-f dockerfiles/bento.dockerfile "
            ". "
            f"--progress={progress}"
        ),
        echo=True,
        pty=not WINDOWS,
    )


@task
def start_bento(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker run --rm --platform=linux/amd64 -p 3000:3000 bento-app:latest",
        echo=True,
        pty=not WINDOWS,
    )
