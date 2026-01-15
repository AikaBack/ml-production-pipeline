import sys
import os

# 1. Сначала жестко задаем путь к проекту, чтобы Python сразу его увидел
PROJECT_PATH = '/home/aikaback/ml_prod_pipeline'
if PROJECT_PATH not in sys.path:
    sys.path.insert(0, PROJECT_PATH)

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd

# 2. Теперь загружаем остальные настройки из .env
load_dotenv(os.path.join(PROJECT_PATH, '.env'))

# 3. Импортируем нашу логику
try:
    from scripts.ml_logic import calculate_psi, run_retraining
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    # Это поможет нам увидеть в логах Airflow, куда он смотрел
    print(f"PYTHONPATH: {sys.path}")

def check_drift_task():
    ref = pd.read_csv(os.path.join(PROJECT_PATH, 'data/reference_data.csv'))
    curr = pd.read_csv(os.path.join(PROJECT_PATH, 'data/current_data.csv'))
    
    psi = calculate_psi(ref.iloc[:, 0], curr.iloc[:, 0])
    print(f"--- МОНИТОРИНГ: PSI Score = {psi} ---")
    return psi > 0.2

with DAG(
    'ml_prod_pipeline',
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False
) as dag:

    monitor_drift = ShortCircuitOperator(
        task_id='monitor_data_drift',
        python_callable=check_drift_task
    )

    retrain_model = PythonOperator(
        task_id='retrain_and_register_model',
        python_callable=run_retraining
    )

    monitor_drift >> retrain_model
