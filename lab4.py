import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_curve,
    auc
)

# 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
print("=" * 60)
print("1. Загрузка обработанных датасетов (из Лабы №1)")
print("=" * 60)

train_df = pd.read_csv("processed_train.csv")
test_df = pd.read_csv("processed_test.csv")

# Воссоздаем целевую переменную 'room_type_Entire home/apt'
for df_item in [train_df, test_df]:
    if 'room_type_Entire home/apt' not in df_item.columns:
        df_item['room_type_Entire home/apt'] = (
            (df_item['room_type_Private room'] == 0) & 
            (df_item['room_type_Shared room'] == 0)
        ).astype(int)

target_cls_col = 'room_type_Entire home/apt'
room_type_cols = [col for col in train_df.columns if col.startswith('room_type_')]

X_train = train_df.drop(columns=room_type_cols)
y_train = train_df[target_cls_col]

X_test = test_df.drop(columns=room_type_cols)
y_test = test_df[target_cls_col]

print(f"Размер обучающей выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")


# 2. МЕТОД СЛУЧАЙНОГО ЛЕСА (Random Forest & OOB Error)
print("\n" + "=" * 60)
print("2. Метод случайного леса (Random Forest) с оценкой OOB")
print("=" * 60)

# RandomForestClassifier с активированной оценкой Out-Of-Bag (oob_score=True)
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

acc_rf = accuracy_score(y_test, y_pred_rf)
oob_score = rf_model.oob_score_
oob_error = 1.0 - oob_score

print(f"Accuracy на тестовой выборке: {acc_rf:.4f}")
print(f"OOB Accuracy (оценка на неиспользованных при бутстрепе данных): {oob_score:.4f}")
print(f"OOB Error (OOBE): {oob_error:.4f}")


# 3. AdaBoost И ГРАДИЕНТНЫЙ БУСТИНГ
print("\n" + "=" * 60)
print("3. Обучение моделей AdaBoost и Gradient Boosting")
print("=" * 60)

# AdaBoost
ada_model = AdaBoostClassifier(n_estimators=100, random_state=42)
ada_model.fit(X_train, y_train)

y_pred_ada = ada_model.predict(X_test)
y_proba_ada = ada_model.predict_proba(X_test)[:, 1]
acc_ada = accuracy_score(y_test, y_pred_ada)

# Градиентный бустинг (Gradient Boosting)
gb_model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
gb_model.fit(X_train, y_train)

y_pred_gb = gb_model.predict(X_test)
y_proba_gb = gb_model.predict_proba(X_test)[:, 1]
acc_gb = accuracy_score(y_test, y_pred_gb)

print(f"Accuracy AdaBoost:           {acc_ada:.4f}")
print(f"Accuracy Gradient Boosting:  {acc_gb:.4f}")


# 4. СРАВНИТЕЛЬНЫЙ АНАЛИЗ И ROC-КРИВЫЕ
print("\n" + "=" * 60)
print("4. Расчет ROC-AUC и построение ROC-кривых")
print("=" * 60)

# Считаем координаты ROC-кривой и ROC-AUC для всех трех моделей
models_proba = {
    'Random Forest': y_proba_rf,
    'AdaBoost': y_proba_ada,
    'Gradient Boosting': y_proba_gb
}

plt.figure(figsize=(8, 6))

for name, proba in models_proba.items():
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc = auc(fpr, tpr)
    print(f"ROC-AUC ({name}): {roc_auc:.4f}")
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Случайная модель')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR)', fontsize=12)
plt.title('Сравнение ROC-кривых ансамблевых моделей', fontsize=14)
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curves_lab4.png')

print("\nГрафик сравнения ROC-кривых успешно сохранен как 'roc_curves_lab4.png'")
