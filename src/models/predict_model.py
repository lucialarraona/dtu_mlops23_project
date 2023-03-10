
import os
import random
import sys

import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv
from huggingface_hub import notebook_login
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
#from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          BertForSequenceClassification, BertTokenizerFast,
                          Trainer, TrainingArguments)

#sys.path.append(os.getcwd())
#print(sys.path.append(os.getcwd()))

sys.path.insert(1, os.path.join(sys.path[0], ".."))

import argparse
import logging
import os
from pathlib import Path

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.preprocessing import LabelEncoder

# 
import wandb
from data.make_dataset import TextDataset  # import our dataset class
from get_project_root import root_path

#notebook_login()

log = logging.getLogger(__name__)
torch.cuda.empty_cache() # for better performance

@click.command()
@click.argument("model_filepath") 
@click.argument("data_filepath", type=click.Path())


def main(model_filepath, data_filepath):
    """
    Returns the loss and accuracy for a given data using a pretrained network.
            Parameters:
                    model_filepath (string): path to a pretrained model
                    data_filepath (string): path to raw data from a random user
    """
    #wandb.init(project='dtu_mlops', 
    #        entity='lucialarraona',
    #        name="inference-2",
    #        #tags=["baseline", "low-lr", "1epoch", "test"],
    #        group='bert-inference',
    #        #config = config, #specify config file to read the hyperparameters from 
    #        )

    wandb.init(mode="disabled")

    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument("load_model_from", default="")
    # add any additional argument that you want
    args = parser.parse_args(sys.argv[2:])
    print(args)

    max_length = 200
    #tokenizer = BertTokenizerFast.from_pretrained(model_filepath, do_lower_case=True)
    #model = BertForSequenceClassification.from_pretrained(model_filepath,# Use the 12-layer BERT model, with an uncased vocab.
                                                       # num_labels = 6, # The number of output labels--2 for binary classification.  si pones num_labels=1 hace MSE LOSS
                                                       # output_attentions = False, # Whether the model returns attentions weights. True, for future visualization
                                                       # output_hidden_states = False ,# Whether the model returns all hidden-states.   
                                                       # vocab_size=tokenizer.vocab_size)

    # lucixls/models
    tokenizer = AutoTokenizer.from_pretrained(model_filepath)
    model = AutoModelForSequenceClassification.from_pretrained(model_filepath)                                                  



    # Load raw data from a random sample input
    sample_raw_data = pd.read_csv(data_filepath, ';', header = None, names = ['text', 'emotion'])
    # Coding the text labels to numbers for training 
    labelencoder = LabelEncoder()
    # Assigning numerical values and storing in another column
    sample_raw_data['emotion_cat'] = labelencoder.fit_transform(sample_raw_data['emotion'])

    le_name_mapping = dict(zip(labelencoder.classes_, labelencoder.transform(labelencoder.classes_)))
    print(le_name_mapping) # Check which category is assigned to what class
    
    X_sample = sample_raw_data['text']
    Y_sample = sample_raw_data['emotion_cat']

    # Convert to our dataset format 
    sample_encodings = tokenizer(sample_raw_data.text.values.tolist(),
    add_special_tokens=True,
    truncation=True,
    max_length=max_length,
    return_token_type_ids=False,
    padding=True,
    return_attention_mask=False,
    return_tensors='pt')
    # Convert our sample tokenized data into a torch Dataset
    sample_dataset = TextDataset(sample_encodings, torch.from_numpy(Y_sample.values))


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

    #project_root = root_path(ignore_cwd=False)

     # Access data from processed folder (for definition of the trainer object)
    #train_dataset = torch.load(project_root + '/data/processed/train.pth') 
    #valid_dataset = torch.load(project_root + '/data/processed/valid.pth')
    
    project_root = Path(__file__).parent.parent.parent
    train_dataset = torch.load(project_root.joinpath('data', 'processed', 'train.pth'))
    valid_dataset = torch.load(project_root.joinpath('data', 'processed', 'valid.pth'))


    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        overwrite_output_dir = True,
        num_train_epochs=3,              # total number of training epochs
        evaluation_strategy="epoch",
        save_strategy = 'epoch',
        per_device_train_batch_size=32,   # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        learning_rate = 5e-5,
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
        metric_for_best_model = 'accuracy',
                                            
                                            # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
        logging_steps=400,               # log & save weights each logging_steps
        save_steps=400,
        #report_to="wandb"                # report to WANDB to keep track of the metrics :) 
    )

    trainer = Trainer(
        model=model,                         # the instantiated Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=valid_dataset,           # evaluation dataset
        compute_metrics=compute_metrics,     # the callback that computes metrics of interest
    )


    trainer.evaluate()

    # Obtain predictions on test set (trainer.predict())
    predictions,labels, metrics = trainer.predict(sample_dataset)  
    # Conf matrix definition
    matrix = confusion_matrix(labels, predictions.argmax(axis=1))
    
    cm_df = pd.DataFrame(matrix,
                     index = ['anger','fear','joy', 'love','sadness','surprise'], 
                     columns = ['anger','fear','joy', 'love','sadness','surprise'])

    # Confusion matrix with counts (plot)
    plt.figure(figsize = (10,7))
    sns.set(font_scale=1.0)
    sns.heatmap(cm_df, annot=True, cmap='Blues',fmt='g')
    plt.xlabel("Predicted class")
    plt.ylabel("True class") 
    plt.savefig(project_root.joinpath('reports', 'figures', 'cfm_predict.png'))



    # Classification report
    clas_report = classification_report(labels, predictions.argmax(axis=1))
    print(clas_report)
    print(metrics)
    None


#??For an interactive model where you can input text and and it will tell you the emotion (future)
"""
# Define the target names 
# target_names= ['sad','joy',]
def get_prediction(text):
    # prepare our text into tokenized sequence
    inputs = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to("cuda")
    # perform inference to our model
    outputs = model(**inputs)
    # get output probabilities by doing softmax
    probs = outputs[0].softmax(1)
    # executing argmax function to get the candidate label
    return target_names[probs.argmax()]
"""

if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    load_dotenv(find_dotenv())

    main()