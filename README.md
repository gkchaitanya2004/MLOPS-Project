
**Name :** Krishna Chaitanya Gorle

**Roll No :** DA25M011

## 1. SetUp
Download prometheus,alertmanger and node expoter from there websites

### 1.1 Airflow


Before running the **dags.py**. We need to install **airflow** and **docker**.Its recommended to install via docker.Here is a tutorial for installing

[Docker Installation](https://docs.docker.com/engine/install/ubuntu/)
[Airflow Installation](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html)
[MlFlow Installation](https://mlflow.org/docs/latest/ml/tracking/quickstart/)




(Replace their docker compose yaml with mine innside the airflow)

After doing this now place the `dags.py`  inside dags folder.

Now add these line in `docker_compose.yaml`
```yaml
    AIRFLOW__EMAIL__EMAIL_BACKEND: airflow.utils.email.send_email_smtp
    AIRFLOW__SMTP__SMTP_HOST: smtp.gmail.com
    AIRFLOW__SMTP__START_TLS: True
    AIRFLOW__SMTP__SMTP_SSL: False
    AIRFLOW__SMTP__SMTP_USER: YOUR_EMAIL_ID
    AIRFLOW__SMTP__SMTP_PASSWORD: 16 DIGIT PASSWORD
    AIRFLOW__SMTP__SMTP_PORT: 587
    AIRFLOW__SMTP__SMTP_MAIL_FROM: YOUR_EMAIL_ID

```
This is required for **Part-B** for your run and to create the password refer here:

[Email Alerts with Airflow](https://medium.com/@chibuokejuliet/email-alerting-with-airflow-c0a5a1f413b4)


Now run 
```bash
docker-compose up --build -d (inside airflow folder)
```

you can go to `localhost:8080` and can see **Airflow** login page and the **user id** and **password** is typically **airflow**

### 1.2 MLFLOW Setup 

## 🏃‍♀️  Run

1) Start the server

```bash
mlflow server --port 5000
```

2) Run the following code to start the **MLFlow Server**

```bash
mlflow run . -e model --experiment-name="CNN Cat vs Dog Classification"
```
Parameters allowed

```
- Learning Rate 
- Batch Size
- Number of Epochs
- Dropout
- Hidden Units (Only 4 values check in model.py)
```

If you **don't pass these parameters** model takes the default values. If you wish to pass the params please run this

```bash
mlflow run . -e model --experiment-name="CNN Cat vs Dog Classification" -P learning_rate=<lr> -P batch_size=<batch_size> -P num_epochs=<epochs> -P dropout=<p>
```

3) This trigger the training of the model and you can track the **metrics** at `https://localhost:5000` This will open the **MLflow UI**, and please ensure that you are in **model training** part instead of **genai** in the ui you can check that in top in the side bar

4) Its recommended to **register** the model.To do that please do the following:
```
MLFLow UI -> Experiments -> Models -> Choose Final Mdel -> Register the Model
```

## 1.3 Final App SetUp
In the root dir run.

```bash
docker compose up -d --build
```

### 2 List of Services


* **FastAPI**
  URL: http://localhost:8000
  Description: Handles model inference and exposes the `/predict` endpoint.

* **Streamlit (User Interface)**
  URL: http://localhost:8501


* **Prometheus (Monitoring System)**
  URL: http://localhost:9090

* **Grafana (Visualization Dashboard)**
  URL: http://localhost:3000


* **Alertmanager (Alert System)**
  URL: http://localhost:9093


* **Node Exporter (System Metrics Collector)**
  URL: http://localhost:9100

## 3 Good Practices
- Add `dashboard.json ` in grafana while you cerating a new dashboard

- Actuall we should do `dvc remote add -d <remote name> <remote url>` and `dvc push` so people can pull it using `dvc pull` which i dont have any remote so dat is placed here at the g drive link:
    https://drive.google.com/drive/folders/1mpvbI1PG2arzwNMm_mjehiiiB51NxMiO?usp=sharing

but you can locally set using that cmd

- For alerts firing using `Alert Manager` use MailTrap and use that username and password in `alertsmanager.yml `

