import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import re
import os

import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import joblib
import re
import os

## 1 Define paths
te_dir = 'Data/Raw/Telugu/'
te_prec_dir = 'Data/Preprocessed/Telugu/'
te_train_pth = te_dir + 'te_train.csv'

## 2 Load data
te_train = pd.read_csv(te_train_pth)

## 3 Preprocess data

def preprocess_te_data(te_train):

    ### 3.1 Remove unnecessary columns
    te_train_prec = te_train.drop(columns=['S.No'])

    ### 3.2 Remove rows with missing values
    te_train_prec = te_train_prec.dropna()

    ### 3.3 Rename columns
    te_train_prec = te_train_prec.rename(columns={'Label':'labels','Comments':'text'})

    ### 3.4 Remove duplicates
    te_train_prec = te_train_prec.drop_duplicates()

    ### 3.5 remove too much spacing in between the words
    te_train_prec['text'] = te_train_prec['text'].replace(r'\s+', ' ', regex=True)

    return te_train_prec

## 4 Encode labels

def encode_labels_te(te_train_prec):
    label_encoder = LabelEncoder()
    te_train_prec['labels'] = label_encoder.fit_transform(te_train_prec['labels'])

    return te_train_prec, label_encoder

## 5 Save the preprocessed data and label encoder
os.makedirs(te_prec_dir, exist_ok=True)
os.makedirs('Models', exist_ok=True)

te_train_prec = preprocess_te_data(te_train)
te_train_prec, label_encoder = encode_labels_te(te_train_prec)

joblib.dump(label_encoder, 'Models/te_label_encoder.joblib')
te_train_prec.to_csv(te_prec_dir + 'te_prec.csv', index=False)
