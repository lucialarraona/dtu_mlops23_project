# MLOps Final Project - Emotion Analysis of Text 🤔🤬😄

- Lucía Larraona (s220492)
- Zuzanna Rowinska (s220351)
- Karol Charylo (s220243)


============================== Updates (3rd week) ==================================
## 👍🏽 Use our deployed model interactively! 

- Navigate to HuggingFace Hub and use the hosted inference API.
https://huggingface.co/lucixls/models

- Or use the following command (changing the "user text to test" with your own text):

`curl -m 310 -X POST https://europe-west1-mlops-374314.cloudfunctions.net/mlops-project -H "Content-Type: application/json" -d '{"text": "user text to test"}'`


## ⛓Our MLOps pipeline (an overview)

![](reports/figures/overview_mlops.png)


============================== Project Proposal (1st week) =========================

## 💡 Project Goal

The aim of this project is to perform multiclass classification on different sentences to identify the emotion. 

## ⚙️ Model and Framework intented

We decided to use the Transformers framework in Pytorch, since our problem falls in the category of Natural Language Processing. 

Our plan is to utilize the strength of the Transformers framework, which provides thousands of pretrained models to perform different tasks. As a starting point we intend to use the pretrained BERT or distilBERT transformer and fine-tune for classification in our data.


## 📊 Data

- We are using the Kaggle dataset [Emotions dataset for NLP](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp). 
- The data is already divided into Train, Validation and Test. 
- Each sample has the following information: a *sentence* (text), and an *emotion* (label) which can be one out of the following 6 categories: sadness, anger, fear, joy, love, and surprise. 


We chose this dataset because of its simplicity, and for the given timeframe the extension of the problem and its implementation is feasible.


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
