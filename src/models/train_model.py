
import torch
#from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import  TrainingArguments, Trainer
from transformers import Trainer, TrainingArguments
import numpy as np
import random
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from data.make_dataset import TextDataset # import our dataset class 

# 
import wandb
import os
import logging 

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


log = logging.getLogger(__name__)

# Initiate wandb logging
wandb.init(project='dtu_mlops', 
           entity='lucialarraona',
           name="bert-test",
           #tags=["baseline", "low-lr", "1epoch", "test"],
           group='bert')


def main():

    # Access data from processed folder
    train_dataset = torch.load("data/processed/train.pth")
    valid_dataset = torch.load("data/processed/valid.pth")
    test_dataset = torch.load('data/processed/test.pth')


    # ---------------- Model Definition / Tokenization / Encoding / Metrics definition ---------------------

    # Define model-name (based on hugging-face library)
    model_name = "bert-base-uncased"
    
    # Hugging Face has its own tokenizer for the transformer: Load the tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=True)
    
    # -------- Training with Trainer function from HuggingFace
    # Load the model and pass to CUDA

    model = BertForSequenceClassification.from_pretrained(model_name,# Use the 12-layer BERT model, with an uncased vocab.
                                                        num_labels = 6, # The number of output labels--2 for binary classification.  si pones num_labels=1 hace MSE LOSS
                                                        output_attentions = False, # Whether the model returns attentions weights.
                                                        output_hidden_states = False ,# Whether the model returns all hidden-states.   
                                                        vocab_size=tokenizer.vocab_size)

    # Check device 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(device)

    # Define metrics for evaluating the classification model and pass it to the Trainer object
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }


    training_args = TrainingArguments(
        output_dir='models/',          # output directory /models directory
        overwrite_output_dir = True,    # will overwrite the last trained model (so it doesnt build up)
        num_train_epochs=3,              # total number of training epochs
        evaluation_strategy='epoch',
        save_strategy = 'epoch',
        per_device_train_batch_size=64,   # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        learning_rate = 0.0005,
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='models/logs',            # directory for storing logs
        load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
        metric_for_best_model = 'accuracy',
                                            
                                            # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
        logging_steps=400,               # log & save weights each logging_steps
        save_steps=400,
        report_to='wandb'                # report to WANDB to keep track of the metrics :) 
    )

    trainer = Trainer(
        model=model,                         # the instantiated Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=valid_dataset,           # evaluation dataset
        compute_metrics=compute_metrics,     # the callback that computes metrics of interest
    )

    log.info("Start training...")
    trainer.train() # start the training
    
    log.info("Finish! :D")

    # Save the model and tokenizer for predict_model
    model_path = '/models/models_trained'
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)




    # ------------------------------ Evaluation of the model----------------

    # WE DONT CALL TRAINER.TRAIN() we call TRAINER.EVALUATE()
    trainer.evaluate()

    # Obtain predictions on test set (trainer.predict())
    predictions,labels, metrics = trainer.predict(test_dataset)  
    # Conf matrix definition
    matrix = confusion_matrix(labels, predictions.argmax(axis=1))

    # Confusion matrix with counts (plot)
    plt.figure(figsize = (10,7))
    sns.set(font_scale=2.0)
    sns.heatmap(matrix, annot=True, cmap='Reds',fmt='g')
    plt.xlabel("Predicted class")
    plt.ylabel("True class") 
    plt.savefig('/reports/figures/cfm_train.png')

    # Classification report
    clas_report = classification_report(labels, predictions.argmax(axis=1))
    print(clas_report)
    print(metrics)
    None



if __name__ == "__main__":
    main()

