# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

Where you instead should add your answers. Any other changes may have unwanted consequences when your report is
auto-generated at the end of the course. For questions where you are asked to include images, start by adding the image
to the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

Will generate a `.html` page of your report. After the deadline for answering this template, we will auto-scrape
everything in this `reports` folder and then use this utility to generate a `.html` page that will be your serve
as your final hand-in.

Running

```bash
python report.py check
```

Will check your answers in this template against the constraints listed for each question e.g. is your answer too
short, too long, or have you included an image when asked. For both functions to work you mustn't rename anything.
The script has two dependencies that can be installed with

```bash
pip install typer markdown
```

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
* [x] Remember to fill out the `requirements.txt` and `requirements_dev.txt` file with whatever dependencies that you
    are using (M2+M6)
* [x] Remember to comply with good coding practices (`pep8`) while doing the project (M7)
* [x] Do a bit of code typing and remember to document essential parts of your code (M7)
* [x] Setup version control for your data or part of your data (M8)
* [x] Add command line interfaces and project commands to your code where it makes sense (M9)
* [x] Construct one or multiple docker files for your code (M10)
* [x] Build the docker files locally and make sure they work as intended (M10)
* [x] Write one or multiple configurations files for your experiments (M11)
* [x] Used Hydra to load the configurations and manage your hyperparameters (M11)
* [ ] Use profiling to optimize your code (M12)
* [x] Use logging to log important events in your code (M14)
* [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [x] Consider running a hyperparameter optimization sweep (M14)
* [x] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

### Week 2

* [x] Write unit tests related to the data part of your code (M16)
* [x] Write unit tests related to model construction and or model training (M16)
* [x] Calculate the code coverage (M16)
* [x] Get some continuous integration running on the GitHub repository (M17)
* [ ] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
* [x] Add a linting step to your continuous integration (M17)
* [x] Add pre-commit hooks to your version control setup (M18)
* [x] Add a continues workflow that triggers when data changes (M19)
* [ ] Add a continues workflow that triggers when changes to the model registry is made (M19)
* [x] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
* [ ] Create a trigger workflow for automatically building your docker images (M21)
* [x] Get your model training in GCP using either the Engine or Vertex AI (M21)
* [x] Create a FastAPI application that can do inference using your model (M22)
* [x] Deploy your model in GCP using either Functions or Run as the backend (M23)
* [x] Write API tests for your application and setup continues integration for these (M24)
* [ ] Load test your application (M24)
* [ ] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [x] Create a frontend for your API (M26)

### Week 3

* [x] Check how robust your model is towards data drifting (M27)
* [ ] Deploy to the cloud a drift detection API (M27)
* [x] Instrument your API with a couple of system metrics (M28)
* [ ] Setup cloud monitoring of your instrumented application (M28)
* [ ] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [x] If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [x] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

* [ ] Write some documentation for your application (M32)
* [ ] Publish the documentation to GitHub Pages (M32)
* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [x] Create an architectural diagram over your MLOps pipeline
* [x] Make sure all group members have an understanding about all parts of the project
* [x] Uploaded all your code to GitHub

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

55

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

s154097, s250393, s251116

### Question 3
> **A requirement to the project is that you include a third-party package not covered in the course. What framework**
> **did you choose to work with and did it help you complete the project?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

We chose to use `uv` for package dependency management in our project. We wanted to compare it with `poetry`, a dependency management tool with which we were familiar. We found several advantages with `uv`, including significant speedups during dependency resolution and package installation, as it is built in Rust. This efficiency was particularly beneficial for maintaining rapid development cycles.

Another notable advantage of `uv` was its ability to handle isolated builds effectively. This was especially helpful for installing complex packages like `torch` plugins, such as `flash-attention`, which often require intricate build processes. With `uv`, we avoided common issues with `poetry` and saved valuable time during dependency installation. These features made `uv` a crucial tool in completing our project.



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

We used `uv` to manage dependencies in our project. The list of dependencies was specified in a `uv` configuration file, which ensures deterministic builds by pinning exact versions of all packages. This made sharing and replicating our development environment straightforward for team members.

To simplify setup further, we included a development container (`dev container`) in our project. The dev container is preconfigured to automatically set up a development environment with `uv` installed and ready to use. A new team member would need to open the project in a compatible IDE, such as Visual Studio Code, which would detect the dev container configuration and spin up the environment.

This approach ensures consistency, avoids compatibility issues, and provides a seamless onboarding experience for new contributors.

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. What did you fill out? Did you deviate from the template in some way?**
>
> Recommended answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
>
> Answer:

We initialized the repository using the MLOps cookie-cutter template developed by Nicki. As described previously, we decided to use uv instead of requirement files. We have added a .dvc file with a config file for data versioning with pointers to the GCP bucket that holds our preprocessed dataset. We have also removed the tasks.py and instead specify project scripts in the pyproject.toml which is compatible with uv. Using uv ensures the environment is always correctly set up when running defined scripts.
Other than that the general project structure is the same as illustrated by the README.md file. We use the src/project structure where all components are developed in the project folder. Dockerfiles are in /dockerfiles, all dependencies are handled using uv and the pyproject.toml in the root of the project.

### Question 6

> **Did you implement any rules for code quality and format? What about typing and documentation? Additionally,**
> **explain with your own words why these concepts matters in larger projects.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used ... for linting and ... for formatting. We also used ... for typing and ... for documentation. These*
> *concepts are important in larger projects because ... . For example, typing ...*
>
> Answer:

We used Ruff for linting to ensure good-quality code and consistent formatting. Our pre-commit checks were set up not to allow commits if the files were not formatted correctly. The Ruff linter automatically enforces the use of docstrings for proper function documentation. We generally used type-hints throughout the code and have used Pydantic for specific modules, improving the readability of the code. This is important for larger projects with multiple collaborators where many components are built by separate developers before being combined in the final application.
We additionally implemented pre-commit checks to ensure that new code passed checks before being merged into the protected main branch. This is important as we could otherwise encounter compatibility issues between different components (given that the tests are correctly written). And even though we took these precautions, we ran into a variety of compatibility issues, which was more likely due to the fact we did not create a clear enough project description before we started developing.

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

We have implemented 18 tests in total. These tests include testing the API endpoints, model training configurations, frontend interactions, model behavior, and data preprocessing functions. They cover some of the most critical parts of our application, including tensor manipulation, configuration validation, service communication, model training steps, and data handling processes.

### Question 8

> **What is the total code coverage (in percentage) of your code? If your code had a code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

Whenever a push was made to the main branch, or a pull request was made to main or dev, we generated a code coverage report. The total code coverage at the latest PR to main is 35%. While we do not have 100% coverage of our code (and that coverage is really quite low), our tests cover some of the most critical parts of our project, including the model and training functions. Further coverage could be implemented with the addition of more tests, especially for inference and data processing. 

Even if one had developed tests such that one achieved code coverage of 100%, this still would not ensure any undesired bugs or implementation errors in the code. This is because bugs might exist outside the data and scenarios specifically tested. 

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

We made use of branches and PRs in our projects. Our project was structured with a protected ‘main’ branch and an unprotected ‘dev’ branch from which we generated ‘feature’ branches whenever a new feature had to be developed. Each group member would then create a feature branch for features such as data preprocessing or model training and merge with the ‘dev’ branch. The ‘dev’ branch was occasionally merged with the ‘main’ branch. The ‘main’ branch protection required at least one reviewer before merging the code. We also implemented a GitHub workflow, ensuring that committed code passed all defined tests and printing dataset statistics if a new dataset was created.

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

We used DVC in the data preprocessing component of our project, although primarily as a proof-of-concept as we are working with a static dataset. It was primarily implemented, as the functionality would be good if one continued developing the application with more continuous machine learning in mind. As our project concerned fine-tuning a ModernBERT model on the MMLU dataset and creating a simple classifier based on that dataset, data-versioning is not a critical feature (as the dataset is static). If the project were to evolve into continuous learning with dynamic data streams, DVC would become essential for tracking dataset changes and their impact on model performance. For example, it would allow us to rollback to previous versions if new data introduced bias or degraded performance, compare model results across different dataset versions, and maintain reproducibility as the data evolves. DVC would also enable better collaboration by letting team members synchronize their local datasets with a remote storage while keeping the actual data separate from the Git repository. We did however struggle quite a lot with managing dvc caches and the creation and loading of old/newer datasets across machines.

### Question 11

> **Discuss you continuous integration setup. What kind of continuous integration are you running (unittesting,**
> **linting, etc.)? Do you test multiple operating systems, Python  version etc. Do you make use of caching? Feel free**
> **to insert a link to one of your GitHub actions workflow.**
>
> Recommended answer length: 200-300 words.
>
> Example:
> *We have organized our continuous integration into 3 separate files: one for doing ..., one for running ... testing*
> *and one for running ... . In particular for our ..., we used ... .An example of a triggered workflow can be seen*
> *here: <weblink>*
>
> Answer:

Our continuous integration (CI) setup is organized into two main GitHub Actions workflows: one for data validation and another for running tests. The data validation workflow, defined in .github/workflows/cml_data.yaml, is triggered on pull requests to the main and dev branches when changes are made to .dvc files. This workflow includes steps for checking out the code, installing the necessary dependencies using uv, authenticating with Google Cloud Platform, pulling data with DVC, and generating data statistics reports. It also comments on the pull request with the generated report.
The testing workflow in .github/workflows/tests.yaml is triggered on pushes and pull requests to the main branch. It employs a matrix strategy to test across multiple operating systems (Ubuntu, Windows, and macOS) and uses Python 3.12. This workflow installs the project dependencies and runs unit tests using pytest.
Both workflows utilize caching to speed up dependency installation, enhancing efficiency. This CI setup ensures our code is continuously validated and tested across different environments, maintaining high code quality.

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

We configured experiments using Hydra to manage configuration files and Pydantic for data validation, providing a flexible and robust setup. The configuration is defined in YAML files and loaded using Hydra, allowing easy management and version control of different experiment settings. Pydantic models validate and structure the configuration data, ensuring type safety. This setup allows for seamless integration of complex configurations and simplifies running experiments with different parameters. For example, we can run an experiment:
```python
uv run train experiment=exp1
```
Where exp1 is a yaml configuration in the experiment folder overriding training input arguments.

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

To ensure the reproducibility of our experiments, we utilize configuration files managed by Hydra. Whenever an experiment is run, the configuration parameters, including hyperparameters and dataset paths, are stored in a structured YAML format. This allows us to track and modify settings for each experiment easily. Additionally, we log all relevant metadata, such as the version of the code, the environment, and the specific configurations used, ensuring that no information is lost. To reproduce an experiment, one would simply need to run the command:
```python
uv run my_experiment.py --config-path=configs --config-name=experiment_config.yaml
```
This command retrieves the exact configuration used in the original experiment, allowing for consistent results. Furthermore, we leverage DVC (Data Version Control) to manage datasets and model versions, ensuring that the data used in experiments is also versioned and accessible. This comprehensive approach guarantees that our experiments can be reliably reproduced, facilitating collaboration and validation of results.

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

![Wandb-1](https://github.com/bertramhojer/mlops-55/blob/main/reports/figures/wandb_1.png)
The first image is a screenshot of example loss graphs. These graphs inform us if the model performance is improving throughout the training, if overfitting is occurring (seen as a reduction in validation performance as the train loss continues decreasing), and can offer other clues about what may be going on in the training process, which can assist with troubleshooting. For example, the noisy but relatively linear training loss curve indicates that the model is not learning from the data, and further model testing and development is needed to determine if there is an issue with the model or the data.

![Wandb-2](https://github.com/bertramhojer/mlops-55/blob/main/reports/figures/wandb_2.png)
The second image is a screenshot of the changing parameters from a W&B sweep, where we tested a variety of hyperparameters including optimizer type, learning rate, and batch size, which are important for model optimization. The sweep used Bayesian optimization, which utilizes the information from earlier sweep runs to select the next model’s hyperparameters and achieve our goal (minimize validation loss). As the models were not achieving strong performance, there was unfortunately limited value in performing the sweep at this time.

![Wandb-3](https://github.com/bertramhojer/mlops-55/blob/main/reports/figures/wandb_3.png)
The third screenshot shows some testing metrics that have been measured and recorded. This includes a confusion matrix, which tells us how well a model is about to identify each class correctly/incorrectly, and a table including several traditional classification metrics, which indicate how well the model performs and generalizes to the test set. 

All wandb results can be found at: https://wandb.ai/mlops_55 

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

For our project, we developed several Docker images to streamline our workflow, including:

Dev Container: We use a dev container for a consistent development environment. This container can be run using Visual Studio Code's Remote Containers extension to ensure all team members have the same setup.
Preprocessing Image: This image handles data preprocessing tasks, including dataset creation and DVC setup. It ensures that data is processed consistently and stored correctly in our cloud storage.
API: The API image serves our machine learning models via a FastAPI application, allowing us to deploy them as scalable web services.
Deployment images: we run the frontend applications, including FastAPI and Streamlit. They provide a user-friendly interface for interacting with our models and visualizing results. The API image serves our machine learning models via a FastAPI application, allowing us to deploy them as scalable web services, while the frontend image serves the Streamlit application.

Using Docker ensures that our applications run consistently across different environments, making development, testing, and deployment more efficient. Docker containers encapsulate all dependencies, configurations, and code, reducing the "it works on my machine" problem and facilitating seamless collaboration.

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

All group members employed the standard VS Code debugger (or Cursor, a fork of VS Code) for debugging code. We did not do any profiling during development of our code, but doing so would have been valuable for identifying performance bottlenecks and optimizing resource usage. This is especially true as some of us had limited access to computational resources, and ran out of memory when training bigger models or using larger batch sizes. 

Debugging locally was easy, but we ran into issues when attempting to debug code when deploying to the cloud. As we worked on training models using GCP (Engine & Vertex AI) we ran into a lot of issues regarding permissions and correct use of various environment variables even though we attempted to use secrets to store said variables. This debugging was slow because it often involved building and pushing docker images and running them in the cloud to debug them.



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

In our project, we mainly used the GCP Bucket and Cloud Run (and partially a basic Compute Engine VM Instance). We used Bucket and DVC to store versions of each dataset, raw and processed. The raw data was used to create dataset statistics, and the processed data was used to train the actual model. Our data preprocessing script automatically preprocesses data and pushes it to DVC and the defined GCP Bucket. We attempted to use Vertex AI for model training, but could not successfully get our models training there due to GPU limitations. We instead opted to train our model using a distributed approach on an HPC to which we had access.  Still, we implemented the pipeline to train a model using CPU on the Compute Engine (which was rather slow). We also deployed a very simple Cloud Run application using FastAPI and Streamlit which loaded a model from our Weights and Biases model registry to be used by an end-user.

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

As explained in Q17, we initially attempted to use the GCP Compute Engine for model training, but ultimately opted for an HPC due to GCP compute limitations. However, we successfully used the Compute Engine for our initial application deployment before transitioning to Cloud Run.

Our deployment architecture employed supervisord to orchestrate multiple containers within a single VM instance, enabling the use of the FastAPI backend and Streamlit frontend services. This setup provided a cost-effective solution for our proof-of-concept phase, as the e2-medium instance offered a balanced compromise between performance and resource allocation. The use of supervisord simplified our container management by allowing us to define and control multiple processes through a single configuration file, eliminating the need for complex container orchestration solutions like Kubernetes for our initial deployment. We transitioned to using Cloud Run as this was a better fit for the simple application we deployed and is a pay-by-use solution.

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

Our GCP bucket contains versions of our dataset. In data.py you can call the load_from_dvc(file) function with e.g. file = ‘mmlu’ which will load a datasetConfig with a train, validation and test split of the preprocessed MMLU dataset (or use the dockerfile for building a training image).
![GCP Bucket](https://github.com/bertramhojer/mlops-55/blob/main/reports/figures/gcp-bucket.png)



### Question 20

> **Upload 1-2 images of your GCP artifact registry, such that we can see the different docker images that you have**
> **stored. You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

![Artifact Registry](https://github.com/bertramhojer/mlops-55/blob/main/reports/figures/registry.png)


### Question 21

> **Upload 1-2 images of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

![Cloud Build History](https://github.com/bertramhojer/mlops-55/blob/main/reports/figures/cloud-build.png)


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

We attempted to use Vertex AI for model training due to is obvious pros such as cloud bucket integration, managed ML workflows, and containerized environment support. While we successfully set up a basic CPU-based training pipeline on Vertex AI using our Docker configuration, we encountered limitations in accessing GPU resources despite having approved GPU quotas. Given the computational demands of our model, CPU-based training was not a viable option even though we did implement the pipeline to run a simple training using the CPU. We thus ended up training our model on an available HPC infrastructure instead. While it would have been great to use a tool such as Vertex AI we simply ran into too many issues given our time and monetary constraints. Resource management and resource availability can be quite a pain. We thus used an alternative that we were lucky to have access to.

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

We did manage to write an API for our model. We implemented the API by defining endpoints for health checks, testing, and predictions. The /predict endpoint is crucial. It takes a query and list of choices and processes them using the fine-tuned ModernBERT model, returning the predicted probabilities for each answer choice.

We additionally used Pydantic for data validation to ensure incoming requests conform to the expected structure. We also implemented an asynchronous lifespan context manager to load the model and tokenizer when the application starts and clean them when it shuts down. This optimized resource management and ensured that the model was ready for inference as soon as the API was running.

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

We deployed an API both locally and to GCP using FastAPI. The app runs on a standard E2 instance with only a CPU. Using CPU only is fine as we are deploying a relatively small model. If working only with the API and hosting it locally, it can be called with the following curl command:
```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{
   "query": "What is the capital of France?",
   "choices": ["Paris", "London", "Berlin", "Madrid"]
}'
```
We additionally deployed the simple API and Streamlit frontend to Cloud Run. See our response to Q28 for more details on that implementation.

[Streamlit Frontend](figures/streamlit.png)


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

We performed very simple unit testing of the API and did not perform any load testing of the API. We utilized pytest for all other tests and used unittest.mock to create a mock registry for the API. The mock functionality allowed us to simulate the behavior of the model and tokenizer without requiring the actual implementations. We wrote a single test case for the predicted endpoint, which is the critical feature of the API, which checks that the response status code is 200 and contains the expected fields. With the latest test of the api we did however not successfully manage to mock the weights and biases handling, and thus have no tests for the final version that was deployed to Cloud Run.

We could have used a tool like Locust to simulate multiple concurrent users for load testing. This allowed us to assess how the API would perform under stress (probably not very well) and identify any potential bottlenecks. Load testing is an essential component of deploying larger systems and provides insights into how much an API can handle before performance starts to degrade.

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

We did not implement monitoring of the deployed model, in terms of measuring model decay over time. This is because our project was within the natural language processing domain, and we used a single dataset. To modify this task, we implemented monitoring for data drift between our training and test sets using Evidently and some parts of this tutorial. More specifically, we obtained the hidden states from both datasets using the deployed model, pooled them with attention masks to get full sentence embeddings, and assessed drift detection by testing whether a binary classifier could successfully determine each datapoint’s original dataset. This allowed us to evaluate how well the model generalized embeddings across data (sub)sets.

However, this implementation does not directly monitor the deployed model or real-time production data. To do this, we would need to use the same drift reporting process to compare new datasets against ours.


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

We ended up using below 5$ in credits in total for the project as we didn’t manage to run any proper training using GCP which would most certainly have been the most expensive part. The main use-cases in terms of the GCP for our project was using the GCP Bucket for data storage as well as deploying a simple app using Cloud Run. Storage is incredibly cheap and Cloud Run is a very cheap option in terms of cloud solutions when using smaller machines (and when there is no traffic to your application).

If we could have gotten our cloud setup working properly and had gotten access to GPUs such that we could have trained using e.g. Vertex AI that would certainly have driven up our costs for developing the project.

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

As discussed in Q24, we implemented a front end for our API and deployed it using Cloud Run. We built two docker images, one for the api and one for the frontend and pushed them to the GCP registry. Those two containers are then used as the foundation for the Cloud Run app. The API and the Frontend can continuously be updated by running (for the api):
```
docker build -t gcr.io/$PROJECT_ID/modernbert-api -f dockerfiles/api.dockerfile . 
docker push gcr.io/$PROJECT_ID/modernbert-api
gcloud run deploy modernbert-api --image gcr.io/$PROJECT_ID/modernbert-api
```
We can do the same for the frontend which will automatically update the Cloud Run instance.
The backend loads the most recent model from a specified project in Weights & Biases, where trained models are saved in a model registry.

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

![Diagram](https://github.com/bertramhojer/mlops-55/blob/main/reports/figures/diagram.png)

The starting point of the diagram is our local setup, where we integrated Docker, DVC, and Wandb into our code. The development environment is managed using DevContainer, which ensures consistency across different setups. The core application includes modules for Model, Train, Hyperparameter optimization, Data, Frontend, API, and Visualize, with dependencies managed via UV and specified in pyproject.toml. Dependencies are versioned with a uv.lock which we push to GitHub. Data is versioned using DVC, which links local datasets to cloud storage, ensuring efficient tracking and reproducibility.

Whenever we commit code and push to GitHub, it automatically triggers GitHub Actions, which handle continuous integration and deployment workflows. These actions build the application, run tests, and push artifacts to Google Artifact Registry, ensuring that the latest versions of models and dependencies are readily available for deployment.

From there, the diagram shows how the cloud infrastructure is structured. The Compute Engine and Vertex AI services in Google Cloud Platform (GCP) retrieve artifacts from the registry and perform computational tasks, such as training machine learning models. The trained models and processed data are stored in a Wandb Artifact Registry, which acts as a central repository for deployment-ready assets. Once the models are validated, they are deployed to production via WandB, ensuring scalable and efficient inference.

Wandb is used to track experiments and visualize model performance, helping to optimize training processes. Additionally, a dedicated Debugger is included in the local environment to facilitate real-time analysis and issue resolution.

In summary, our workflow ensures integration between local development and cloud deployment. It leverages GitHub for version control, CI/CD automation through GitHub Actions, and GCP services for scalable compute and storage. This setup allows us to iterate fast, maintain reproducibility, and deploy robust machine learning models efficiently.


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

We faced a lot of challenges throughout the course in general and the development of our project. As Nicki mentioned in his final lecture data processing is a major hurdle that wasn’t really covered in this course and that isn’t part of the standard MLOps diagram, it is however something we spent a lot of time on. 

On a related note, our results indicate that the model is not learning from the data properly, and the loss does not improve during the training process. This may be due to many things. For example, the dataset includes a wide variety of subject matters, which may not have been represented fairly in the splits. Further investigation and troubleshooting of both the model and the dataset was needed, but not prioritized, as this course was intended to practice the implementation and usage of MLOps tools. 
We additionally spent quite a lot of time trying to get training working using GCP. We couldn't access GPUs on Vertex AI despite approved quotas, forcing us to switch to HPC infrastructure. We also struggled with GCP bucket permissions when writing model artifacts, which required careful IAM policy configuration.

We also attempted to do proper data versioning with DVC, but that ended up presenting us with more challenges than it did us favors. Establishing consistent versioning protocols and remote storage configurations took considerable effort to standardize and in the end we are not certain we got it working properly.

Ensuring compatibility between the different components developed by contributors was also quite a challenge. We are referring to components such as preprocessing, model training and e.g. the frontend application. While docker helped us standardize development environments (via e.g. the use of a devcontainer) and we used uv to manage our environment and dependencies in general we still ran into a lot of issues.

We tried to integrate various continuous integration workflows such as the ones described in Q11, but we still ran into a lot of issues. This has highlighted the need to spend a lot of time preparing and designing a project or an application before the actual coding begins. We are painfully aware that using a more test-driven development approach where it is clearly defined what our program must be capable of before starting development would have been incredibly useful.



### Question 31

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
>
> Answer:

Student s154097 was responsible for developing the model and training scripts using pytorch lightning as well as training the model using Vertex AI and deploying them using the Compute Engine and Artifact Registry. They additionally developed the devcontainer and development setup.

Student s250393 was in charge of setting up the github repository and the cookie cutter project. They additionally developed that data processing and configured the dvc and GCP Bucket configuration as well as developed and deployed the API and frontend. 

Student s251116 developed aspects of the training script and all evaluation/visualization and logging (including WANDB logging and model sweeps). They also implemented coverage calculations, continuous workflow, and data drifting functionality. 

All code was developed in a collaborative manner, including bug bashing and additional list items not mentioned, such that every group member has an idea of how every component of the entire pipeline works. The above contribution statement is thus just an indicator of who was the primus motor for the specified parts of the project.
