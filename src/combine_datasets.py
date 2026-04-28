import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import joblib

## 1 Define paths
hi_prec_dir = 'Data/Preprocessed/Hindi/'
hi_prec_pth = hi_prec_dir + 'hi_prec.csv'

te_prec_dir = 'Data/Preprocessed/Telugu/'
te_prec_pth = te_prec_dir + 'te_prec.csv'

## 2 Load preprocessed data
hi_train_prec = pd.read_csv(hi_prec_pth)
te_train_prec = pd.read_csv(te_prec_pth)

## 3 Add language column to both datasets 
languages = ['Hindi', 'Telugu']
le = LabelEncoder()
labels = le.fit_transform(languages)
hi_train_prec['language'] = labels[0]
te_train_prec['language'] = labels[1]

## 4 Combine the datasets
combined_train_prec = pd.concat([hi_train_prec, te_train_prec], ignore_index=True)

## 5 Shuffle the combined dataset
combined_train_prec = combined_train_prec.sample(frac=1, random_state=42).reset_index(drop=True)

## 6 Save the combined dataset
combined_prec_dir = 'Data/Preprocessed/'
combined_prec_pth = combined_prec_dir + 'combined_prec.csv'
combined_train_prec.to_csv(combined_prec_pth, index=False)

## 7 save the label encoder 
le_pth = 'Models/' + 'lang_label_encoder.joblib'
joblib.dump(le, le_pth)


