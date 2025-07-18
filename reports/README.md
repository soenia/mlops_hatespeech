## Overall project checklist

The checklist is *exhaustive* which means that it includes everything that you could do on the project included in the
curriculum in this course. Therefore, we do not expect at all that you have checked all boxes at the end of the project.
The parenthesis at the end indicates what module the bullet point is related to. Please be honest in your answers, we
will check the repositories and the code to verify your answers.

### Week 1

* [x] Create a git repository (M5)
* [x] Make sure that all team members have write access to the GitHub repository (M5)
* [x] Create a dedicated environment for you project to keep track of your packages (M2)
* [x] Create the initial file structure using cookiecutter with an appropriate template (M6)
* [x] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
* [x] Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)
* [ ] Remember to fill out the `requirements.txt` and `requirements_dev.txt` file with whatever dependencies that you
    are using (M2+M6)
* [ ] Remember to comply with good coding practices (`pep8`) while doing the project (M7)
* [ ] Do a bit of code typing and remember to document essential parts of your code (M7)
* [x] Setup version control for your data or part of your data (M8)
* [x] Add command line interfaces and project commands to your code where it makes sense (M9)
* [x] Construct one or multiple docker files for your code (M10)
* [x] Build the docker files locally and make sure they work as intended (M10)
* [ ] Write one or multiple configurations files for your experiments (M11)
* [ ] Used Hydra to load the configurations and manage your hyperparameters (M11)
* [ ] Use profiling to optimize your code (M12)
* [x] Use logging to log important events in your code (M14)
* [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [ ] Consider running a hyperparameter optimization sweep (M14)
* [ ] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

### Week 2

* [x] Write unit tests related to the data part of your code (M16)
* [x] Write unit tests related to model construction and or model training (M16)
* [x] Calculate the code coverage (M16)
* [x] Get some continuous integration running on the GitHub repository (M17)
* [x] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
* [ ] Add a linting step to your continuous integration (M17)
* [x] Add pre-commit hooks to your version control setup (M18)
* [ ] Add a continues workflow that triggers when data changes (M19)
* [ ] Add a continues workflow that triggers when changes to the model registry is made (M19)
* [x] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
* [ ] Create a trigger workflow for automatically building your docker images (M21)
* [ ] Get your model training in GCP using either the Engine or Vertex AI (M21)
* [ ] Create a FastAPI application that can do inference using your model (M22)
* [ ] Deploy your model in GCP using either Functions or Run as the backend (M23)
* [ ] Write API tests for your application and setup continues integration for these (M24)
* [ ] Load test your application (M24)
* [ ] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [ ] Create a frontend for your API (M26)

### Week 3

* [ ] Check how robust your model is towards data drifting (M27)
* [ ] Deploy to the cloud a drift detection API (M27)
* [ ] Instrument your API with a couple of system metrics (M28)
* [ ] Setup cloud monitoring of your instrumented application (M28)
* [ ] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [ ] If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [ ] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

* [ ] Write some documentation for your application (M32)
* [ ] Publish the documentation to GitHub Pages (M32)
* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Create an architectural diagram over your MLOps pipeline
* [ ] Make sure all group members have an understanding about all parts of the project
* [ ] Uploaded all your code to GitHub

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**

Group A

### Question 2
> **Enter the study number for each member in the group**

12371375, 12590611

### Question 3
> **A requirement to the project is that you include a third-party package not covered in the course. What framework**
> **did you choose to work with and did it help you complete the project?**
>
> Recommended answer length: 100-200 words.

We used the widely known and popular Huggingface package `transformers` as third-party package in our project. We used functionality for loading pretrained models and tokenizers from the package to do text classification in our project. The package also provided us with a lot of documentation and examples which helped us to quickly get started with our project. Additionally, we used the datasets package from Huggingface to load the dataset we used for training our model. The big variety of models enabled us to use a lightweight model that was able to run on our local machines without the need for a GPU.

## Coding environment

> In the following section we are interested in learning more about you local development environment. This includes
> how you managed dependencies, the structure of your code and how you managed code quality.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Recommended answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:

We used pip for managing our dependencies. We managed dependencies using a combination of a `pyproject.toml` file and separate requirements.txt files, which we have continuously updated during the project: The `requirements.txt` file contains the packages needed to run the code, the `requirements_dev.txt` file contains the packages needed for development (e.g. ruff, precommit) and the `requirements_test.txt` file contains the packages needed for testing (e.g. pytest).The `pyproject.toml` includes project dynamic dependency declarations that reference our requirements.txt and requirements_dev.txt files. This setup allows us to build and install the project with setuptools in a clean and standardized way.

To reproduce our exact development environment, one would have to run the following commands:

```bash
conda create -n hatespeech_env python=3.11
conda activate hatespeech_env
pip install -e .
```

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. What did you fill out? Did you deviate from the template in some way?**
>
> Recommended answer length: 100-200 words


From the cookiecutter template we have filled out the `src`, `reports`, `data` and `tests` folder. We have removed the `docs` folder and the `notebooks` folder because we did not use any documentation or notebooks in our project. We also added a `cloud` folder, which contains yaml files with cloudbuild configurations and vertex AI configurations. The most relevant deviation from the template is that we have added a `logs` folder that contains trained model checkpoints and evaluation results.


### Question 6

> **Did you implement any rules for code quality and format? What about typing and documentation? Additionally,**
> **explain with your own words why these concepts matters in larger projects.**
>
> Recommended answer length: 100-200 words.
>
> Answer:

We used `ruff` for linting. Additionally we did typing and we documented our code consistently. These concepts are important in larger projects because they help to maintain code quality and readability. They are also helpful to better understand the code and to avoid bugs.

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Recommended answer length: 50-100 words.
>
> Answer:

We implemented 4 automatic tests in our code which we divided in two categories: unit tests and integration tests. The unit tests are testing the data loading and preprocessing and the the model training. The integration tests are testing wether the API of the backend (app.py) is working properly and as intended. The tests are run automatically using continuous integration on GitHub whenever we create a pull request or commit code to the main branch. The tests can also run locally using pytest.
Furthermore we wrote a load test script (locustfile.py) which can simulate a load test of multiple users accessing the API at the same time. This script is not run automatically but can be run locally by any group member.

### Question 8

> **What is the total code coverage (in percentage) of your code? If your code had a code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Recommended answer length: 100-200 words.
>
> Answer:

The total code coverage of our code is 37% since some parts of our code are not covered by tests. We are far from 100% coverage of our code and even if we were then we would not trust it to be error free. The reason for this is that code coverage only measures how much of the code is executed during the tests, but it does not measure wether the code is error free or not. So it's possible to have a high code coverage, but still have bugs in the code.

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Recommended answer length: 100-200 words.
>
> Answer:

We made use of both branches and pull requests in our project. In particular, we created for each task an issue on GitHub which contains a check list of smaller parts of the task. The issues were then assigned to a group member who would then create a branch and link the issue to the branch. When the task was completed, the group member would create a pull request to merge the branch into the main branch. The continuous integration would then run tests. The pull request should only be merged if all tests passed. This way we wanted to ensure that the main branch was always in a working state. We could have improved this workflow further by requiring that another group member reviewed the pull request before merging it. Moreover we could have protected the main branch such that it could not be merged into without a reviewed pull request or without successfully passing the continuous integration tests.

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Recommended answer length: 100-200 words.
>
> Answer:

We used DVC for tracking our data (`processed.dvc`) and model checkpoints (`run1.dvc`). Some of our containerized scripts (train.py, data.py and evaluate.py) use dvc pull and push in the corresponding yaml files to automatically pull the data and model checkpoints from the DVC remote storage. This way we can ensure that the data and model checkpoints are always in sync with the code.

### Question 11

> **Discuss you continuous integration setup. What kind of continuous integration are you running (unittesting,**
> **linting, etc.)? Do you test multiple operating systems, Python  version etc. Do you make use of caching? Feel free**
> **to insert a link to one of your GitHub actions workflow.**
>
> Recommended answer length: 200-300 words.
>
> Answer:

We have organized our continuous integration into 2 separate files: one for doing unit testing and one for running linting. For the unit testing we used pytest and for the linting we used ruff. We also made use of caching to speed up the continuous integration and we use multiple operating systems (Linux, MacOS and Windows). We also implemented a workflow that triggers when data changes.
Moreover, we added code coverage to our unit tests using the `coverage` package. The code coverage is then reported to Codecov and also a part of the continuous integration workflow with GitHub Actions.

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: Python  my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

The most relevant instance here is our train.py since it contains a lot of hyperparameters. We used a config file for reproducibility (see next section). We have configured the training script such that it can be run with a command line interface. The following command would run the training script with 10 epochs:

```bash
train --epochs 10
```

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

We configured our experiments using Hydra, which allowed us to manage hyperparameters for our training script via a yaml config file. This made it easy to reproduce experiments and switch between configurations.

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Recommended answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

![wandb](figures/ROC.png)
![wandb2](figures/wandb.png)
We were tracking the ROC curve.


### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments/project? Include how you would run your docker images and include a link to one of your docker files.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

For our project we developed 6 docker images: one for data preprocessing, one for training, one for evaluation, two for the deployment backends (app.dockerfile and bento.dockerfile) and one for the frontend. Note that our actual application uses the app.dockerfile for deployment, but we also created a bento.dockerfile for testing purposes. For the latter we have a very slim image which even contains an onnx model.

For instance, have a look at our at our BentoML App which is lightweight (small container). Note that this is only the backend.
To pull the docker images, you can use the following command:

```bash
docker pull --platform=linux/amd64 europe-west1-docker.pkg.dev/mlops-hatespeech/hs-images/bento-app:latest
```

To run the docker image on a Mac, you can use the following command:

```bash
docker run --rm \
  --platform=linux/amd64 \
  -p 3000:3000 \
  europe-west1-docker.pkg.dev/mlops-hatespeech/hs-images/bento-app:latest
```

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

We primarily used logging and print statements to debug our code.

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

We made use of the following GCP services:
- **Cloud Storage**: Store data, reports, model weights, user requests
- **Artifact Registry**: Store all docker images
- **Cloud Build**: Build docker images and trigger workflows
- **Vertex AI**: Train and deploy models
- **Cloud Run**: Deploy the API
- **Secret Manager**: Store secrets like the WANDB_API_KEY

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:
We used custom Vertex AI Jobs which internally rely on Google Cloud Compute Engine. For this we implemented containerized jobs, pulled from the artifact registry. Note that we also abused those Vertex AI Jobs to run our data preprocessing and evaluation scripts.

Machine Type: n1-highmem-2
We specified a service-account eligible for Vertex AI and Artifact Registry access.


### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

![bucket](figures/GCPbucket.png)
![bucket2](figures/bucket2.png)

### Question 20

> **Upload 1-2 images of your GCP artifact registry, such that we can see the different docker images that you have**
> **stored. You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

![registry](figures/registry.jpg)

### Question 21

> **Upload 1-2 images of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

![build](figures/cloudbuild.jpg)

### Question 22

> **Did you manage to train your model in the cloud using either the Engine or Vertex AI? If yes, explain how you did**
> **it. If not, describe why.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We managed to train our model in the cloud using the Engine. We did this by ... . The reason we choose the Engine*
> *was because ...*
>
> Answer:

Our model training can be executed hierarchically, meaning that we use the same train.dockerfile for the cloud as for local execution. Since the training requires a WANDB_API_KEY environment variable, the vertex AI job a two-stage process with one stage including a placeholder in the vertex config file and the other stage starting a temporary alpine container injecting the secret from the GCP secret manager. The second stage then runs the training script with the WANDB_API_KEY environment variable set.

## Deployment

### Question 23

> **Did you manage to write an API for your model? If yes, explain how you did it and if you did anything special. If**
> **not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did manage to write an API for our model. We used FastAPI to do this. We did this by ... . We also added ...*
> *to the API to make it more ...*
>
> Answer:

We have two apps for demonstration purposes: One uses FastAPI and the other uses BentoML. The FastAPI app is implemented in `app.py` and the BentoML app is implemented in `bento_app.py`. The FastAPI app is a simple API that takes a text input and returns a prediction of whether the text is hateful or not. We also added a frontend for the FastAPI app using Streamlit, which allows users to interact with the API in a more user-friendly way. Both apps are containerized using Docker and can be deployed in the cloud using Cloud Run. The FastAPI app is also able to log the predictions to the Cloud Bucket relevant for data drift detection.

### Question 24

> **Did you manage to deploy your API, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

We deployed our app locally using Docker and in the cloud using Cloud Run. The local deployment can be done by running the following command (assuming you have the docker image pulled with the command shown above):

```bash
docker run --rm \
  --platform=linux/amd64 \
  -p 3000:3000 \
  europe-west1-docker.pkg.dev/mlops-hatespeech/hs-images/bento-app:latest
```

With the above command, the app starts locally.

Try the backend API with the following link:
https://bento-app-178847025464.europe-west1.run.app/

Note that it might a while since our deployment is request based and a Kaltstart is required.


### Question 25

> **Did you perform any unit testing and load testing of your API? If yes, explain how you did it and what results for**
> **the load testing did you get. If not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For unit testing we used ... and for load testing we used ... . The results of the load testing showed that ...*
> *before the service crashed.*
>
> Answer:

We performed unit testing of our API using pytest and load testing using Locust. The unit tests can be run using the command `pytest`.

The results of load testing showed that the API does not crash under high load, but the average response time slightly increases when the number of users increases.
![locust](figures/loadtest.png)

### Question 26

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

We implemented prometheus monitoring of our deployed model.

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 27

> **How many credits did you end up using during the project and what service was most expensive? In general what do**
> **you think about working in the cloud?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ... . Working in the cloud was ...*
>
> Answer:

One group member forgot to stop a virtual machine, which led to higher costs.
The Google Cloud Trial does not include GPU resources, so we couldn't use one of the services.


### Question 28

> **Did you implement anything extra in your project that is not covered by other questions? Maybe you implemented**
> **a frontend for your API, use extra version control features, a drift detection service, a kubernetes cluster etc.**
> **If yes, explain what you did and why.**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We implemented a frontend for our API. We did this because we wanted to show the user ... . The frontend was*
> *implemented using ...*
>
> Answer:

We implemented a frontend for our app that allows users to interact with the API in a more user-friendly way. The frontend is implemented using Streamlit and is able to send requests to the API and display the results. We also added a drift detection service that monitors the requests including input and prediction of the API and logs them to the Cloud Bucket. This allows us to detect if the data is drifting over time. The drift detection looks at the embeddings, keywords and class frequencies of the requests and compares them to the training data.


### Question 29

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally, in your own words, explain the**
> **overall steps in figure.**
>
> Recommended answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and push to GitHub, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

--- question 29 fill here ---

### Question 30

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Recommended answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

Debugging the cloud was one of the biggest challenges in the project, as well as secret management for the cloud. One team member could not build docker images locally, which made it difficult to test the code locally.

### Question 31

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project. Additionally, state if/how you have used generative AI**
> **tools in your project.**
>
> Recommended answer length: 50-300 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
> *We have used ChatGPT to help debug our code. Additionally, we used GitHub Copilot to help write some of our code.*
> Answer:


Student 12371375 set up the continuous integration and wrote most of the unit tests. They also calculate the code coverage, Wandb logging and sweeping.
Student 12590611 was in charge of Docker, the cloud setup and the deployment of the APIs.
General documentation and code maintenance was done by both team members.
We have used ChatGPT to help debug our code.
