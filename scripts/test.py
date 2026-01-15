import requests
import random
import pandas as pd

# Читаем наши данные, чтобы брать реальные примеры
df = pd.read_csv('data/current_data.csv').drop('target', axis=1)

for i in range(20):
    sample = df.sample(1).values[0].tolist()
    payload = {"user_id": f"bot_{i}", "features": sample}
    r = requests.post("http://localhost:8000/predict", json=payload)
    print(r.json())