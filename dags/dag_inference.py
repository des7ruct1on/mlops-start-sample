import io
import json
import logging
import pandas as pd
from datetime import datetime, timedelta
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.models import DAG, Variable
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago

MY_BUCKET = Variable.get("S3_BUCKET")
LOGGER = logging.getLogger()
LOGGER.addHandler(logging.StreamHandler())

DEFAULT_ARGS = {
    "owner": "Ivan Lyapin",
    "email": "shad0w2020@mail.ru",
    "email_on_failure": True,
    "email_on_retry": False,
    "retry": 3,
    "retry_delay": timedelta(minutes=3)
}

dag = DAG(
    dag_id="mlops_pred",
    schedule_interval="0 1 * * *",
    start_date=days_ago(2),
    catchup=False,
    tags=["Prediction_class"],
    default_args=DEFAULT_ARGS
)

def task_init():
    LOGGER.info("Started predictions")

def download_pred_upload(model_path: str, X_test_path: str, y_test_path: str, columns_path: str):
    LOGGER.info("Started downloading")

    s3_hook = S3Hook("s3_connector")

    model_file = s3_hook.download_file(key=model_path, bucket_name=MY_BUCKET)
    model = CatBoostClassifier()
    model.load_model(model_file)
    LOGGER.info(f"Model loaded from {model_path}")

    X_test_file = s3_hook.download_file(key=X_test_path, bucket_name=MY_BUCKET)
    y_test_file = s3_hook.download_file(key=y_test_path, bucket_name=MY_BUCKET)
    columns_file = s3_hook.download_file(key=columns_path, bucket_name=MY_BUCKET)
    LOGGER.info(f"Downloaded {X_test_path}, {y_test_path}, {columns_path}")

    X_test = pd.read_csv(X_test_file)
    y_test = pd.read_csv(y_test_file).squeeze() 
    with open(columns_file, 'r', encoding='utf-8') as file:
        columns_dict = json.load(file)

    test_pool = Pool(data=X_test, cat_features=columns_dict['categorial_cols'])

    pred_proba = model.predict_proba(test_pool)[:, 1]  
    pred = model.predict(test_pool)  

    roc_auc = roc_auc_score(y_test, pred_proba)
    LOGGER.info(f"ROC-AUC: {roc_auc}")

    predictions_df = pd.DataFrame({
        "predictions": pred,
        "probabilities": pred_proba
    })
    file_tmp = io.BytesIO()
    predictions_df.to_csv(file_tmp, index=False)
    file_tmp.seek(0)

    s3_hook.load_file_obj(
        file_obj=file_tmp,
        key="pred.csv",
        bucket_name=MY_BUCKET,
        replace=True
    )
    LOGGER.info("Predictions uploaded to S3")

task_init = PythonOperator(
    task_id="task_init",
    python_callable=task_init,
    dag=dag
)

download_pred_upload_task = PythonOperator(
    task_id="download_pred_upload",
    python_callable=download_pred_upload,
    op_kwargs={
        "model_path": "mlflow/124574803600152026/035024bd41c74b12bc18a18d0065e030/artifacts/CatBoost/model.cb",
        "X_test_path": "Employee_data/X_test.csv",
        "y_test_path": "Employee_data/y_test.csv",
        "columns_path": "Employee_data/columns.json"
    },
    dag=dag
)

task_init >> download_pred_upload_task