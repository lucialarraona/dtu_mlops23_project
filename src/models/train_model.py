
# hello from hpc?
# hello again
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          BertForSequenceClassification, BertTokenizerFast,
                          Trainer, TrainingArguments)

from get_project_root import root_path
import parser
from google.cloud import secretmanager


sys.path.insert(1, os.path.join(sys.path[0], ".."))
print(sys.path.insert(1, os.path.join(sys.path[0], "..")))

import logging
import os

import hydra
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from huggingface_hub import notebook_login
from omegaconf import DictConfig
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)

import wandb
from data.make_dataset import TextDataset  # import our dataset class

#os.environ["WANDB_DISABLED"] = "true" # disable logging when using cloudbuild 

log = logging.getLogger(__name__)
@hydra.main(config_path="../config", config_name="default_config.yaml") # specify path of config file to later pass it to wandb 

def main(config: DictConfig):

    """
    Returns the loss and accuracy after training the project's model and testing in on test.txt raw data.
    Saves a figure of the confusion matrix for the classification task
    """

    #client = secretmanager.SecretManagerServiceClient()
    #PROJECT_ID = "713387486048"

    #secret_id = "WANDB"
    #resource_name = f"projects/{PROJECT_ID}/secrets/{secret_id}/versions/1"
    #response = client.access_secret_version(name=resource_name)
    #api_key = response.payload.data.decode("UTF-8")
    #os.environ["WANDB_API_KEY"] = api_key
    # Initiate wandb logging
    wandb.init(project='dtu_mlops', 
            entity='lucialarraona',
            tags=["gcp-run"],
            group='bert',
            config = config, #specify config file to read the hyperparameters from 
           )

    #wandb.init(mode="disabled")
            
    
    project_root = Path(__file__).parent.parent.parent
    print(project_root)
    train_dataset = torch.load(str(project_root.joinpath('data', 'processed', 'train.pth')))
    valid_dataset = torch.load(str(project_root.joinpath('data', 'processed', 'valid.pth')))
    test_dataset = torch.load(str(project_root.joinpath('data', 'processed', 'test.pth')))



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
        output_dir='models/',                               # output directory /models directory
        overwrite_output_dir = True,                        # will overwrite the last trained model (so it doesnt build up)
        num_train_epochs= config.train.epochs,              # total number of training epochs
        evaluation_strategy='epoch',
        save_strategy = 'epoch',
        per_device_train_batch_size=config.train.train_batch_size, # batch size per device during training
        per_device_eval_batch_size=config.train.test_batch_size,   # batch size for evaluation
        learning_rate = config.train.lr,
        warmup_steps=500,                                     # number of warmup steps for learning rate scheduler
        weight_decay=config.train.weight_decay,               # strength of weight decay
        logging_strategy= 'epoch',
        logging_dir= str(project_root.joinpath('models', 'logs')),                            # directory for storing logs
        load_best_model_at_end=True,                          # load the best model when finished training (default metric is loss)
        metric_for_best_model = 'accuracy',
                                            
                                                              # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
        logging_steps=400,                                    # log & save weights each logging_steps
        save_steps=400,
        report_to='wandb'                                     # report to WANDB to keep track of the metrics :) 
        #push_to_hub = True,
        #hub_token = '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' # shouldnt be here but oh well
    )

    trainer = Trainer(
        model=model,                         # the instantiated Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=valid_dataset,          # evaluation dataset
        compute_metrics=compute_metrics,     # the callback that computes metrics of interest
    )

    log.info("Start training...")
    trainer.train() # start the training
    
    log.info("Finish! :D")

    # Pushing it to huggingface hub (for posterior donwloading it easier)
    #trainer.push_to_hub()  # push it to the huggingface repository (cloud)
    #tokenizer.push_to_hub("lucixls/models")

    # Save the model and tokenizer for predict_model (locally)
    model.save_pretrained(str(project_root.joinpath('models', 'models_trained')))
    tokenizer.save_pretrained(str(project_root.joinpath('models', 'models_trained')))


    # ------------------------------ Evaluation of the model----------------
    print('Evaluating on test data...')
    trainer.evaluate()

    # Obtain predictions on test set (trainer.predict())
    predictions,labels, metrics = trainer.predict(test_dataset)  
    # Conf matrix definition
    matrix = confusion_matrix(labels, predictions.argmax(axis=1))
    cm_df = pd.DataFrame(matrix,
                     index = ['anger','fear','joy', 'love','sadness','surprise'], 
                     columns = ['anger','fear','joy', 'love','sadness','surprise'])

    # Confusion matrix with counts (plot)
    plt.figure(figsize = (10,7))
    sns.set(font_scale=1.0)
    sns.heatmap(cm_df, annot=True, cmap='Reds',fmt='g')
    plt.xlabel("Predicted class")
    plt.ylabel("True class") 
    plt.savefig(str(project_root.joinpath('reports', 'figures', 'cfm_train.png')))

    # Classification report
    clas_report = classification_report(labels, predictions.argmax(axis=1))
    print(clas_report)
    print(metrics)
    None


if __name__ == "__main__":

    project_dir = Path(__file__).resolve().parents[2]
    load_dotenv(find_dotenv())

    main()

