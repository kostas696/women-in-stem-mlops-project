from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.utils import timezone
from datetime import timedelta

default_args = {
    "owner": "konstantinos",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="women_in_stem_mlops_pipeline",
    default_args=default_args,
    description="End-to-end ML pipeline for Women in STEM project",
    schedule=None, 
    start_date=timezone.utcnow(),
    catchup=False,
) as dag:

    preprocess = BashOperator(
        task_id="preprocess_data",
        bash_command="python3 /home/konstan_souf/mlops/women-in-stem-mlops-project/src/preprocess.py"
    )

    train_rf = BashOperator(
        task_id="train_random_forest",
        bash_command="python3 /home/konstan_souf/mlops/women-in-stem-mlops-project/src/train.py randomforest"
    )

    train_xgb = BashOperator(
        task_id="train_xgboost",
        bash_command="python3 /home/konstan_souf/mlops/women-in-stem-mlops-project/src/train.py xgboost"
    )

    train_cb = BashOperator(
        task_id="train_catboost",
        bash_command="python3 /home/konstan_souf/mlops/women-in-stem-mlops-project/src/train.py catboost"
    )

    evaluate = BashOperator(
        task_id="evaluate_models",
        bash_command="python3 /home/konstan_souf/mlops/women-in-stem-mlops-project/src/evaluate.py"
    )
    
    generate_monitoring_reports = BashOperator(
        task_id="generate_monitoring_reports",
        bash_command="python3 /home/konstan_souf/mlops/women-in-stem-mlops-project/monitoring/generate_report.py"
    )

    # DAG flow
    preprocess >> [train_rf, train_xgb, train_cb] >> evaluate >> generate_monitoring_reports
    
# Register DAG object explicitly
globals()["women_in_stem_mlops_pipeline"] = dag
