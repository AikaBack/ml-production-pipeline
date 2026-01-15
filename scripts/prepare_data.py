import pandas as pd
from sklearn.datasets import load_breast_cancer
import os

def generate_data():
    os.makedirs('data', exist_ok=True)
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    
    # Эталонные данные (Reference)
    ref_data = df.sample(frac=0.5, random_state=42)
    ref_data.to_csv('data/reference_data.csv', index=False)
    
    # Текущие данные (с искусственным дрифтом)
    curr_data = df.drop(ref_data.index)
    # Искусственно меняем значения в одной из колонок
    curr_data['mean radius'] = curr_data['mean radius'] * 1.5 
    curr_data.to_csv('data/current_data.csv', index=False)
    
    print("Данные успешно созданы в папке data/")

if __name__ == "__main__":
    generate_data()