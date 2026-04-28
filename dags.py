
from datetime import datetime, timedelta
from unittest import result
from airflow.models import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.smtp.operators.smtp import EmailOperator
from airflow.utils.email import send_email

import json
import subprocess
import joblib
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, '/opt/airflow')

from evidently import Report
from evidently.presets import DataDriftPreset

from src.hindi_preprocess import preprocess_hi_data 
from src.telugu_preprocess import preprocess_te_data

def load_data(hi_pth, te_pth):
    hi_data = pd.read_csv(hi_pth)
    te_data = pd.read_csv(te_pth)

    os.makedirs("/opt/airflow/Data/Production/Hindi/", exist_ok=True)
    os.makedirs("/opt/airflow/Data/Production/Telugu/", exist_ok=True)
    hi_data.to_csv("/opt/airflow/Data/Production/Hindi/hi_prod.csv", index=False)
    te_data.to_csv("/opt/airflow/Data/Production/Telugu/te_prod.csv", index=False)

def process_data():

    hi_data = pd.read_csv("/opt/airflow/Data/Production/Hindi/hi_prod.csv")
    te_data = pd.read_csv("/opt/airflow/Data/Production/Telugu/te_prod.csv")

    hi_data_prec = preprocess_hi_data(hi_data)
    te_data_prec = preprocess_te_data(te_data)
    hi_encoder = joblib.load('Models/hi_label_encoder.joblib')
    te_encoder = joblib.load('Models/te_label_encoder.joblib')

    hi_data_prec['labels'] = hi_encoder.transform(hi_data_prec['labels'])
    te_data_prec['labels'] = te_encoder.transform(te_data_prec['labels'])

    os.makedirs("/opt/airflow/Data/Production/Preprocessed/Hindi/", exist_ok=True)
    os.makedirs("/opt/airflow/Data/Production/Preprocessed/Telugu/", exist_ok=True)

    hi_data_prec.to_csv("/opt/airflow/Data/Production/Preprocessed/Hindi/hi_prod_prec.csv", index=False)
    te_data_prec.to_csv("/opt/airflow/Data/Production/Preprocessed/Telugu/te_prod_prec.csv", index=False)


def combine_data():
    hi_data_prec = pd.read_csv("/opt/airflow/Data/Production/Preprocessed/Hindi/hi_prod_prec.csv")
    te_data_prec = pd.read_csv("/opt/airflow/Data/Production/Preprocessed/Telugu/te_prod_prec.csv")

    lanaguge_encoder = joblib.load('Models/lang_label_encoder.joblib')
    hi_data_prec['language'] = lanaguge_encoder.transform(['Hindi'])[0]
    te_data_prec['language'] = lanaguge_encoder.transform(['Telugu'])[0]

    combined = pd.concat([hi_data_prec, te_data_prec], ignore_index=True) 
    combined = combined.sample(frac=1)       

    combined.to_csv("/opt/airflow/Data/Production/Preprocessed/combined_prod.csv", index=False)


def check_drift():
        
    train_data = pd.read_csv("/opt/airflow/Data/Preprocessed/combined_prec.csv")
    prod_data = pd.read_csv("/opt/airflow/Data/Production/Preprocessed/combined_prod.csv")

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=train_data, current_data=prod_data)
    result = snapshot.dict()

    drift_val = result["metrics"][0]["value"]["share"]
    print("Drift value:", drift_val)
    if drift_val >= 0.4:
        send_email(
            to="chaitu.gorle@gmail.com",
            subject="Drift Detected",
            html_content="<h2>Drift detected in the production data.Check docs on what to do.</h2>"
        )
        new_data = pd.read_csv("/opt/airflow/Data/Production/Preprocessed/combined_prod.csv")
        train_data = pd.read_csv("/opt/airflow/Data/Preprocessed/combined_prec.csv")
        updated_train_data = pd.concat([train_data, new_data], ignore_index=True)
        updated_train_data.to_csv("/opt/airflow/Data/Preprocessed/combined_prec.csv", index=False)



with DAG(
    'guardrail_pipeline',
    schedule = "@hourly",
    start_date = datetime(2026,4,28),
    max_active_runs = 1,
    catchup = False
) as dag:



    task1 = PythonOperator(
        task_id = "load_data",
        python_callable = load_data,
        op_kwargs = {
            "hi_pth": "/opt/airflow/Data/Raw/Hindi/hi_test.csv",
            "te_pth": "/opt/airflow/Data/Raw/Telugu/te_test.csv"
        }
    )

    task2 = PythonOperator(
        task_id = "process_data",
        python_callable = process_data,
        trigger_rule = "all_done",
    )

    task3 = PythonOperator(
        task_id = "combine_data",
        python_callable = combine_data,
        trigger_rule = "all_done",
    )


    task4 = PythonOperator(
        task_id = "check_drift",
        python_callable = check_drift,
        trigger_rule = "all_done",
    )

    task1 >> task2 >> task3 >> task4
