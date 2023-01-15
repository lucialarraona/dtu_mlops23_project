# -*- coding: utf-8 -*-
import logging
import os
import sys
from pathlib import Path

import click
import pandas as pd
import torch
from dotenv import find_dotenv, load_dotenv
from sklearn.preprocessing import LabelEncoder
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          BertForSequenceClassification, BertTokenizerFast)
from transformers.file_utils import (is_tf_available, is_torch_available,
                                     is_torch_tpu_available)

sys.path.insert(1, os.path.join(sys.path[0], ".."))

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())

def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


    #transform = transforms.Compose(
    #[transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_path = input_filepath + "/train.txt"
    val_path = input_filepath + "/val.txt"
    test_path = input_filepath + "/test.txt"

    df_train = pd.read_csv(train_path, ';', header = None, names = ['text', 'emotion'])
    df_valid = pd.read_csv(val_path, ';', header = None, names = ['text', 'emotion'])
    df_test = pd.read_csv(test_path, ';', header = None, names = ['text', 'emotion'])

    # Coding the text labels to numbers for training 
    labelencoder = LabelEncoder()

#    Assigning numerical values and storing in another column
    df_train['emotion_cat'] = labelencoder.fit_transform(df_train['emotion'])
    df_valid['emotion_cat'] = labelencoder.fit_transform(df_valid['emotion'])
    df_test['emotion_cat'] = labelencoder.fit_transform(df_test['emotion'])

    le_name_mapping = dict(zip(labelencoder.classes_, labelencoder.transform(labelencoder.classes_)))
    print(le_name_mapping) # Check which category is assigned to what class
    # {'anger': 0, 'fear': 1, 'joy': 2, 'love': 3, 'sadness': 4, 'surprise': 5} # result

    X_train = df_train['text']
    Y_train = df_train['emotion_cat']


    X_valid = df_valid['text']
    Y_valid = df_valid['emotion_cat']


    X_test = df_test['text']
    Y_test =  df_test['emotion_cat']


        # Define model-name (based on hugging-face library)
    model_name = "bert-base-uncased"
    # max sequence length for each document/sentence sample 
    max_length = 200

    # Hugging Face has its own tokenizer for the transformer: Load the tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=True)

    # Tokenize the dataset, truncate when passed `max_length`, and pad with 0's when less than `max_length`
    train_encodings = tokenizer(df_train.text.values.tolist(), 
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        return_token_type_ids=False,
        padding=True,
        return_attention_mask=False,
        return_tensors='pt')

    valid_encodings = tokenizer(df_valid.text.values.tolist(), 
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        return_token_type_ids=False,
        padding=True,
        return_attention_mask=False,
        return_tensors='pt')


    test_encodings = tokenizer(df_test.text.values.tolist(),
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        return_token_type_ids=False,
        padding=True,
        return_attention_mask=False,
        return_tensors='pt')


    # Convert our tokenized data into a torch Dataset
    train_dataset = TextDataset(train_encodings, torch.from_numpy(Y_train.values))
    valid_dataset = TextDataset(valid_encodings, torch.from_numpy(Y_valid.values))
    test_dataset = TextDataset(test_encodings, torch.from_numpy(Y_test.values))

    torch.save(train_dataset,output_filepath + "/train.pth" )
    torch.save(valid_dataset,output_filepath + "/valid.pth" )
    torch.save(test_dataset,output_filepath + "/test.pth" )
    

# Create a new dataset with the tokenized input(text) and the labels
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)




if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
