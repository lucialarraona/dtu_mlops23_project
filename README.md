# MLOps Final Project - Emotion Analysis of Text 🤔🤬😄
Repo for the final project of DTU MLOps January course - Jan 2023

- Lucía Larraona (s220492)
- Zuzanna Rowinska (s220351)
- Jakub Solis (s213792)
- Karol Charylo (s220243)

## 💡 Project Goal

The aim of this project is to perform multilabel classification on different sentences to identify the emotion. 

## ⚙️ Model and Framework intented

We decided to use the Transformers framework in Pytorch, since our problem falls in the category of Natural Language Processing. 

Our plan is to utilize the strength of the Transformers framework, which provides thousands of pretrained models to perform different tasks. As a starting point we intend to use the pretrained BERT or distilBERT transformer and fine-tune for classification in our data.


## 📊 Data

- We are using the Kaggle dataset: **Emotions dataset for NLP** https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp . 
- The data is already divided into Train, Vaidation and Test. 
- Each sample has the following information: a sentence (text), and an emotion (label) which can be one out of the following 6 categories: sadness, anger, fear, joy, love, and surprise. 


We chose this dataset because of its simplicity, and for the given timeframe the extension of the problem and its implementation is feasible.

