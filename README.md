deepedge_task
==============================

DeepEdge Deep Learning Task.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── notebooks
         ├── cnn_model.ipynb   <- Approach-1 Simple CNN to predict the non-zero coordinates.
         └── train_flatten.ipynb    <- Approach-2 Flatten input pixels to predict the non-zero coordinates.
    │    
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment.
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code of project available here
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data          
    │   │   └── make_dataset.py   <- Scripts to generate data
    │   │
    │   └── models         
    │       ├── logger.py   <- Scripts to log the model performance.
    │       └── train_cnn.py   <- Scripts to train CNN.
    │
    ├── training_logs    <- Training logs are stored here in Excel form.
    │
    ├── params.yaml    <- All user-defined parameters are defined here. 
    │   
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
