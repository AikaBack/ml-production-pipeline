import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import os

# Путь к логам
PROJECT_PATH = '/home/aikaback/ml_prod_pipeline'
LOG_FILE = os.path.join(PROJECT_PATH, 'logs/ab_test_logs.csv')

def generate_report():
    if not os.path.exists(LOG_FILE):
        print("Ошибка: Файл логов не найден. Сначала отправьте запросы через Flask.")
        return

    df = pd.read_csv(LOG_FILE)
    
    if len(df) < 10:
        print(f"Предупреждение: Мало данных для анализа (всего {len(df)} записей).")

    print("="*50)
    print("ФИНАЛЬНЫЙ ОТЧЕТ ПО РЕЗУЛЬТАТАМ A/B ТЕСТИРОВАНИЯ")
    print("="*50)

    # 1. Анализ трафика
    stats = df['model_version'].value_counts()
    total = len(df)
    print(f"\n[1] РАСПРЕДЕЛЕНИЕ ТРАФИКА:")
    for ver, count in stats.items():
        print(f" - Версия {ver}: {count} запросов ({count/total:.1%})")

    # 2. Анализ предсказаний (Behavior Analysis)
    # Считаем, сколько раз каждая модель предсказала 'Злокачественную опухоль' (0)
    print(f"\n[2] АНАЛИЗ ПРЕДСКАЗАНИЙ (Класс 0 - Злокачественная):")
    
    # Создаем таблицу сопряженности
    contingency = pd.crosstab(df['model_version'], df['prediction'])
    
    for ver in contingency.index:
        pred_0 = contingency.loc[ver, 0] if 0 in contingency.columns else 0
        total_ver = stats[ver]
        rate = (pred_0 / total_ver) * 100
        print(f" - Модель {ver}: Предсказано '0' в {rate:.2f}% случаев")

    # 3. Статистическая проверка (Chi-Square Test)
    # Проверяем, есть ли существенная разница в поведении моделей
    print(f"\n[3] СТАТИСТИЧЕСКАЯ ЗНАЧИМОСТЬ:")
    
    if len(stats) < 2:
        print(" - Недостаточно групп для сравнения (нужны и A, и B).")
    else:
        chi2, p, dof, ex = chi2_contingency(contingency)
        print(f" - P-value: {p:.4f}")
        
        if p < 0.05:
            print(" - РЕЗУЛЬТАТ: Различия СТАТИСТИЧЕСКИ ЗНАЧИМЫ.")
            print("   Внимание: Новая модель ведет себя иначе, чем старая.")
        else:
            print(" - РЕЗУЛЬТАТ: Различия СТАТИСТИЧЕСКИ НЕ ЗНАЧИМЫ.")
            print("   Вывод: Новая модель сохраняет преемственность предсказаний.")

    # 4. Итоговое решение
    print(f"\n[4] ИТОГОВОЕ РЕШЕНИЕ (Decision Support):")
    if p > 0.05:
        print(" >>> РЕКОМЕНДОВАНО: Перевести модель из Staging в Production.")
    else:
        print(" >>> ТРЕБУЕТСЯ ПРОВЕРКА: Новая модель выдает слишком другие результаты.")
    print("="*50)

if __name__ == "__main__":
    generate_report()