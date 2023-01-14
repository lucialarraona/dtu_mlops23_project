import os
import os.path
from cgi import test
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder
from tests import _PATH_DATA


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_raw_data():

    train_path = _PATH_DATA + "/raw/train.txt"
    val_path = _PATH_DATA + "/raw/val.txt"
    test_path = _PATH_DATA + "/raw/test.txt"

    df_train = pd.read_csv(train_path, ';', header = None, names = ['text', 'emotion'])
    df_valid = pd.read_csv(val_path, ';', header = None, names = ['text', 'emotion'])
    df_test = pd.read_csv(test_path, ';', header = None, names = ['text', 'emotion'])

    # Coding the text labels to numbers for training 
    labelencoder = LabelEncoder()

#    Assigning numerical values and storing in another column
    df_train['emotion_cat'] = labelencoder.fit_transform(df_train['emotion'])
    df_valid['emotion_cat'] = labelencoder.fit_transform(df_valid['emotion'])
    df_test['emotion_cat'] = labelencoder.fit_transform(df_test['emotion'])


    X_train = df_train['text']
    Y_train = df_train['emotion_cat']

    X_valid = df_valid['text']
    Y_valid = df_valid['emotion_cat']

    X_test = df_test['text']
    Y_test =  df_test['emotion_cat']

    N_train = 16000
    N_test = 2000
    N_valid = 2000

    assert len(X_train) == N_train, "Dataset did not have the correct number of training samples"
    assert len(X_test) == N_test,  "Dataset did not have the correct number of testing samples"
    assert len(X_valid) == N_valid, "Dataset did not have the correct number of validation samples"