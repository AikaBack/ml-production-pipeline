import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from pycaret.classification import setup, compare_models, finalize_model
from mlflow.tracking import MlflowClient

def calculate_psi(expected, actual, buckets=10):
    def scale(series, bins):
        return pd.cut(series, bins, labels=False, duplicates='drop')
    expected_percents = pd.Series(scale(expected, buckets)).value_counts(normalize=True)
    actual_percents = pd.Series(scale(actual, buckets)).value_counts(normalize=True)
    df = pd.DataFrame({'exp': expected_percents, 'act': actual_percents}).fillna(0.0001)
    return np.sum((df['act'] - df['exp']) * np.log(df['act'] / df['exp']))

def run_retraining():
    PROJECT_PATH = '/home/aikaback/ml_prod_pipeline'
    # Используем абсолютный путь к базе
    mlflow.set_tracking_uri(f"sqlite:///{PROJECT_PATH}/mlflow.db")
    
    data = pd.read_csv(f'{PROJECT_PATH}/data/current_data.csv')
    
    # ВАЖНО: log_experiment=False. Мы не даем PyCaret плодить пустые папки
    s = setup(data=data, target='target', session_id=123, 
              log_experiment=False, verbose=False, html=False)
    
    print("--- Обучение... ---")
    best_model = compare_models()
    final_model = finalize_model(best_model)
    
    # Ручное логирование: один запуск = одна папка с моделью
    print("--- Регистрация в MLflow ---")
    with mlflow.start_run(run_name="Clean_Production_Run") as run:
        run_id = run.info.run_id
        # Явно создаем папку 'model' внутри артефактов
        mlflow.sklearn.log_model(final_model, "model")
        
        # Регистрируем в реестре
        model_name = "production_model"
        model_uri = f"runs:/{run_id}/model"
        client = MlflowClient()
        mv = mlflow.register_model(model_uri, model_name)
        
        # Сразу ставим в Staging
        client.transition_model_version_stage(
            name=model_name, version=mv.version, stage="Staging"
        )
        print(f"--- УСПЕХ: Версия {mv.version} создана в папке {run_id} ---")

if __name__ == "__main__":
    run_retraining()