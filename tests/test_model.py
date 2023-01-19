import os
import os.path
import numpy as np
from cgi import test
import pytest
from tests import _PATH_DATA
import torch
from transformers import (BertForSequenceClassification, BertTokenizerFast)
import pandas as pd
from sklearn.preprocessing import LabelEncoder


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_model():

    val_path = _PATH_DATA + "/raw/val.txt"

    df_valid = pd.read_csv(val_path, ';', header = None, names = ['text', 'emotion'])

    labelencoder = LabelEncoder()

    df_valid['emotion_cat'] = labelencoder.fit_transform(df_valid['emotion'])

    Y_valid = df_valid['emotion_cat']
# Define model-name (based on hugging-face library)
    model_name = "bert-base-uncased"
    
    # Hugging Face has its own tokenizer for the transformer: Load the tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=True)
    
    model = BertForSequenceClassification.from_pretrained(model_name,# Use the 12-layer BERT model, with an uncased vocab.
                                                        num_labels = 6, # The number of output labels--2 for binary classification.  si pones num_labels=1 hace MSE LOSS
                                                        output_attentions = False, # Whether the model returns attentions weights.
                                                         output_hidden_states = False ,# Whether the model returns all hidden-states.   
                                                        vocab_size=tokenizer.vocab_size)

    # Check device 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    assert len(np.unique(Y_valid)) == model.num_labels, 'Number of labels do not match'
    assert device.type == 'cpu' or device.type == 'cuda', 'Could define the device'
