
# Define model-name (based on hugging-face library)
# We are trying 2, bert and distilledbert

import torch
#from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import  TrainingArguments, Trainer
from transformers import Trainer, TrainingArguments
import numpy as np
import random
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

# Hugging Face has its own tokenizer for the transformer: Load the tokenizer

import wandb
import os
os.environ['WANDB_API_KEY'] = ''

# ---------------- Model Definition / Tokenization / Encoding / Metrics definition ---------------------

# Define model-name (based on hugging-face library)
model_name = "bert-base-uncased"
# max sequence length for each document/sentence sample 
max_length = 200

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




# ----------------- Sweep config and run----------------------

# method
sweep_config = {
    'method': 'random'
}

# hyperparameters to try 
parameters_dict = {
    'epochs': {
        'value': [5,10,15,20]
        },
    'batch_size': {
        'values': [8, 16, 32, 64]
        },
    'learning_rate': {
        'distribution': 'log_uniform_values',
        'min': 5e-5,
        'max': 5e-3
    },
    'weight_decay': {
        'values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    },
}


sweep_config['parameters'] = parameters_dict


# start the sweep in Wandb API 
sweep_id = wandb.sweep(sweep_config, 
                        project='dtu_deepl_models', 
                        entity = 'lucialarraona',)



# define training function with the config file parameters as inputs 


def train(config=None): 
  with wandb.init(config=config):
    # set sweep configuration
    config = wandb.config

    # set training arguments
    training_args = TrainingArguments(
        output_dir='./results',
	      report_to='wandb',  # Turn on Weights & Biases logging
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=16,
        logging_dir='./logs',            # directory for storing logs
        metric_for_best_model = 'accuracy',
        save_strategy='epoch',
        evaluation_strategy='epoch',
        logging_strategy='epoch',
        load_best_model_at_end=True,
        remove_unused_columns=False,

    )

    train_dataset = torch.load("../data/processed/test.pth")
    valid_dataset = torch.load("../data/processed/test.pth")

    # define training loop
    trainer = Trainer(
        model=model,                         # the instantiated Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=valid_dataset,           # evaluation dataset
        compute_metrics=compute_metrics,     # the callback that computes metrics of interest
    )

# start training loop
    trainer.train()


wandb.agent(sweep_id, train, count=20)