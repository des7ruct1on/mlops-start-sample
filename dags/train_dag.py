import io
import json
import logging
import os
import pandas as pd

from datetime import datetime, timedelta
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.models import DAG, Variable
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago

_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())
BUCKET = Variable.get("S3_BUCKET")
DEFAULT_ARGS = {
    "owner": "Ivan Lyapin",
    "retry": 3,
    "retry_delay": timedelta(minutes=1)
}
RANDOM_STATE = 42

grid_catboost = {
    'iterations': [100, 200],
    'learning_rate': [0.05, 0.1],
    'depth': [6, 8],
    'l2_leaf_reg': [1, 3],
}

dag = DAG(
    dag_id="train_catboost",
    schedule_interval="0 1 * * *",
    start_date=days_ago(2),
    catchup=False,
    tags=["mlops"],
    default_args=DEFAULT_ARGS
)

def init() -> None:
    _LOG.info("started pipeline")

def prepare_data() -> None:
    _LOG.info("started preparing")

    s3 = S3Hook("s3_connector")

    file = s3.download_file(
        key='heart.csv',
        bucket_name=BUCKET
    )

    df = pd.read_csv(file)
    X = df.drop(['HeartDisease'], axis=1)
    y = df['HeartDisease']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=RANDOM_STATE
    )

    for name, data in zip(
        ["X_train", "X_test", "y_train", "y_test"],
        [X_train, X_test, y_train, y_test]
    ):
        file_tmp = io.BytesIO()
        data.to_csv(file_tmp, index=False)
        file_tmp.seek(0)
        s3.load_file_obj(
            file_obj=file_tmp,
            key=f"{name}.csv",
            bucket_name=BUCKET,
            replace=True
        )

    _LOG.info("preparation finished")

def train_model() -> None:
    _LOG.info("training started")
    s3_hook = S3Hook("s3_connector")

    data = {}
    for name in ["X_train", "X_test", "y_train", "y_test"]:
        file = s3_hook.download_file(
            key=f"{name}.csv",
            bucket_name=BUCKET,
        )
        data[name] = pd.read_csv(file)

    cat_features = []
    for col in data["X_train"].columns:
        if data["X_train"][col].dtype == 'object':
            cat_features.append(col)

    model_catboost = CatBoostClassifier(random_state=RANDOM_STATE, verbose=False, loss_function='Logloss')
    train_pool = Pool(data=data["X_train"], label=data["y_train"], cat_features=cat_features)
    test_pool = Pool(data=data["X_test"], label=data["y_test"], cat_features=cat_features)

    search_results = model_catboost.randomized_search(
        param_distributions=grid_catboost,
        X=train_pool,
        n_iter=50,
        cv=5,
        partition_random_seed=RANDOM_STATE,
        calc_cv_statistics=True,
        search_by_train_test_split=False
    )

    prediction = model_catboost.predict(test_pool)
    result = {
        'roc-auc': roc_auc_score(data["y_test"], prediction),
        'gini': (2 * roc_auc_score(data["y_test"], prediction) - 1) * 100
    }

    date = datetime.now().strftime("%Y_%m_%d_%H")
    file_tmp = io.BytesIO()
    file_tmp.write(json.dumps(result).encode())
    file_tmp.seek(0)
    s3_hook.load_file_obj(
        file_obj=file_tmp,
        key=f"metrics/{date}.csv",
        bucket_name=BUCKET,
        replace=True
    )
    model_filename = f"catboost_model_{date}.cbm"
    model_catboost.save_model(model_filename)

    with open(model_filename, "rb") as model_file:
        s3_hook.load_file_obj(
            file_obj=model_file,
            key=f"models/{model_filename}",
            bucket_name=BUCKET,
            replace=True
        )

    os.remove(model_filename)
    _LOG.info("finished training")

def save_results() -> None:
    _LOG.info("Success")

task_init = PythonOperator(
    task_id="init",
    python_callable=init,
    dag=dag
)

task_prepare = PythonOperator(
    task_id="prepare_data",
    python_callable=prepare_data,
    dag=dag
)

task_train = PythonOperator(
    task_id="train_model",
    python_callable=train_model,
    dag=dag
)

task_save = PythonOperator(
    task_id="save_results",
    python_callable=save_results,
    dag=dag
)

task_init >> task_prepare >> task_train >> task_save