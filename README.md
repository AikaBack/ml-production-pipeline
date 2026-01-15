# ML Production Pipeline: От мониторинга до A/B теста

## Проект реализует полный жизненный цикл ML-модели в продакшене.

### Описание данных
В проекте используется классический медицинский датасет — Breast Cancer Wisconsin (Diagnostic).
Источник данных: Библиотека scikit-learn (load_breast_cancer).
Задача: Бинарная классификация — определение типа опухоли (злокачественная или доброкачественная) на основе характеристик ядер клеток.
Целевая переменная (Target):
0: Malignant (Злокачественная)
1: Benign (Доброкачественная)
Признаки (Features): Всего 30 числовых параметров, включая:
Средний радиус (mean radius)
Текстура (texture)
Периметр (perimeter)
Площадь (area)
Гладкость (smoothness) и др.

### Используемые модели (AutoML)

Для автоматического подбора и обучения моделей используется библиотека PyCaret. 
Пайплайн настроен на сравнение более чем 15 алгоритмов машинного обучения.
В ходе экспериментов PyCaret автоматически сравнивает:

LDA (Linear Discriminant Analysis).

Logistic Regression

Random Forest Classifier

Extra Trees Classifier

LightGBM (Light Gradient Boosting Machine)

XGBoost

### Основные этапы:
1. **Мониторинг Data Drift:** Airflow DAG использует PSI для отслеживания изменений в данных.
2. **AutoML:** Автоматическое переобучение лучшей модели с помощью PyCaret.
3. **Model Registry:** Регистрация и управление стадиями моделей в MLflow (Staging/Production).
4. **Serving & A/B Testing:** Flask API с динамическим роутингом трафика и логированием запросов.

### Стек технологий:
- **Python 3.10**
- **Orchestration:** Apache Airflow 3.0
- **AutoML:** PyCaret
- **Tracking:** MLflow
- **API:** Flask
- **Environment:** WSL2 (Ubuntu 24.04)

## Как запустить:
Проект разработан для работы в среде Ubuntu/WSL2 с использованием Python 3.10.
1. Подготовка системы (Системные зависимости)
Для работы библиотек машинного обучения (LightGBM) в Ubuntu необходимо установить библиотеку OpenMP:

sudo apt update && sudo apt install libgomp1 -y

2. Клонирование и настройка окружения
code
Bash
### Клонирование репозитория
git clone https://github.com/AikaBack/ml_prod_pipeline.git
cd ml_prod_pipeline

### Создание виртуального окружения (обязательно Python 3.10)
python3.10 -m venv venv
source venv/bin/activate

### Установка библиотек
pip install -r requirements.txt
3. Настройка конфигурации (.env)
Создайте в корне проекта файл .env и укажите в нем абсолютный путь к вашей папке проекта:

PROJECT_PATH=/home/ВАШ_ПОЛЬЗОВАТЕЛЬ/ml_prod_pipeline

MLFLOW_TRACKING_URI=http://127.0.0.1:5000

MODEL_NAME=production_model

TRAFFIC_SPLIT=0.5

4. Запуск инфраструктуры

Для работы проекта нужно открыть три терминала:

Терминал 1: MLflow (Сервер трекинга)

source venv/bin/activate

mlflow server --backend-store-uri sqlite:///$(pwd)/mlflow.db --default-artifact-root $(pwd)/artifacts --host 127.0.0.1 --port 5000

Терминал 2: Airflow (Оркестратор)

source venv/bin/activate

export AIRFLOW_HOME=$(pwd)/airflow_home

При первом запуске Airflow сам создаст базу и пользователя в режиме standalone

airflow standalone

После запуска перейдите на http://localhost:8080, логин admin, пароль будет напечатан в консоли, либо в файле simple_auth в папке airflow_home.

5. Подготовка данных и запуск пайплайна

В новом терминале создайте начальные данные:

python scripts/prepare_data.py

В интерфейсе Airflow включите DAG ml_prod_pipeline и нажмите Trigger DAG.
После завершения в MLflow появится Version 1.

6. Настройка A/B теста

В MLflow (http://localhost:5000) назначьте Version 1 стадию Production.

Запустите DAG в Airflow еще раз, чтобы создать Version 2, и назначьте ей стадию Staging.

8. Запуск API-сервиса (Flask)

Терминал 3: Flask Server

source venv/bin/activate

python app/main.py

8. Проверка (Тестирование)

Запускаем файл test.py для выборки, затем final_report.py для вывода результата

python scripts/test.py

python scripts/final_report.py
