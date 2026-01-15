from flask import Flask, request, jsonify
import mlflow.pyfunc
import pandas as pd
import random
import os
import csv
from datetime import datetime

app = Flask(__name__)

# Пути и настройки
PROJECT_PATH = '/home/aikaback/ml_prod_pipeline'
LOG_FILE = f"{PROJECT_PATH}/logs/ab_test_logs.csv"
os.makedirs(f"{PROJECT_PATH}/logs", exist_ok=True)

# Список колонок для модели Breast Cancer
COLUMN_NAMES = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area',
    'mean smoothness', 'mean compactness', 'mean concavity',
    'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error',
    'smoothness error', 'compactness error', 'concavity error',
    'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area',
    'worst smoothness', 'worst compactness', 'worst concavity',
    'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

mlflow.set_tracking_uri(f"sqlite:///{PROJECT_PATH}/mlflow.db")
MODEL_NAME = "production_model"

# ФУНКЦИЯ ЛОГИРОВАНИЯ (Этап 5)
def log_prediction(user_id, version, prediction):
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            # Заголовки для будущего анализа
            writer.writerow(['timestamp', 'user_id', 'model_version', 'prediction'])
        writer.writerow([datetime.now(), user_id, version, prediction])

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    user_id = data.get('user_id', 'unknown')
    
    # Создаем данные для модели
    features = pd.DataFrame([data['features']], columns=COLUMN_NAMES)
    
    try:
        # A/B сплит
        if random.random() < 0.5:
            model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/Staging")
            ver = "B_Staging"
        else:
            model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/Production")
            ver = "A_Production"
            
        prediction = int(model.predict(features)[0])
        
        # ЗАПИСЫВАЕМ В ЛОГ
        log_prediction(user_id, ver, prediction)
        
        return jsonify({
            "status": "success",
            "version": ver,
            "prediction": prediction
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)