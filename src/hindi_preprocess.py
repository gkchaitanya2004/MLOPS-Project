import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import joblib
import re
import os

## 1 Define paths
hi_dir = 'Data/Raw/Hindi/'
hi_prec_dir = 'Data/Preprocessed/Hindi/'
hi_train_pth = hi_dir + 'hi_train.csv'

## 2 Load data
hi_train = pd.read_csv(hi_train_pth)

## 3 Preprocess data

def preprocess_hi_data(hi_train):

    ### 3.1 Remove unnecessary columns
    hi_train_prec = hi_train.drop(columns=['text_id','task_2', 'task_3'])

    ### 3.2 Remove rows with missing values
    hi_train_prec = hi_train_prec.dropna()

    ### 3.3 Rename columns
    hi_train_prec = hi_train_prec.rename(columns={'task_1':'labels'})

    ### 3.4 Remove duplicates
    hi_train_prec = hi_train_prec.drop_duplicates()

    ### 3.5 Remove unnecessary words in text (@usernames, URLs, etc.)
    hi_train_prec['text'] = hi_train_prec['text'].apply(lambda x : re.sub(r'@\w+', '', x))
    hi_train_prec['text'] = hi_train_prec['text'].apply(lambda x : re.sub(r'#\w+', '', x))
    hi_train_prec['text'] = hi_train_prec['text'].apply(lambda x : re.sub(r'http\S+', '', x))  
    hi_train_prec['text'] = hi_train_prec['text'].apply(lambda x : re.sub(r'www\S+', '', x))

    ### 3.6 remove too much spacing in between the words
    hi_train_prec['text'] = hi_train_prec['text'].replace(r'\s+', ' ', regex=True)

    return hi_train_prec

## 4 Encode labels

def encode_labels_hi(hi_train_prec):
    label_encoder = LabelEncoder()
    hi_train_prec['labels'] = label_encoder.fit_transform(hi_train_prec['labels'])

    return hi_train_prec, label_encoder

## 5 Save the preprocessed data and label encoder
os.makedirs(hi_prec_dir, exist_ok=True)
os.makedirs('Models', exist_ok=True)

hi_train_prec = preprocess_hi_data(hi_train)
hi_train_prec, label_encoder = encode_labels_hi(hi_train_prec)

joblib.dump(label_encoder, 'Models/hi_label_encoder.joblib')
hi_train_prec.to_csv(hi_prec_dir + 'hi_prec.csv', index=False)
