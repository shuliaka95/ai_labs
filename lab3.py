import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    classification_report,
    roc_curve,
    auc
)

# 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
print("=" * 50)
print("1. Загрузка обработанных датасетов")
print("=" * 50)

# Используем те же файлы, подготовленные в Лабораторной №1
train_df = pd.read_csv("processed_train.csv")
test_df = pd.read_csv("processed_test.csv")

print(f"Размер обучающей выборки: {train_df.shape}")
print(f"Размер тестовой выборки: {test_df.shape}")


# 2. ЗАДАЧА РЕГРЕССИИ (Дерево решений)
print("\n" + "=" * 50)
print("2. Задача регрессии (прогноз 'price')")
print("=" * 50)

X_train_reg = train_df.drop(columns=['price'])
y_train_reg = train_df['price']

X_test_reg = test_df.drop(columns=['price'])
y_test_reg = test_df['price']

# Инициализация и обучение DecisionTreeRegressor (с ограничением глубины от переобучения)
dt_reg = DecisionTreeRegressor(max_depth=6, random_state=42)
dt_reg.fit(X_train_reg, y_train_reg)

y_pred_reg = dt_reg.predict(X_test_reg)

mse = mean_squared_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_reg, y_pred_reg)

print("Метрики Дерева Регрессии (DecisionTreeRegressor):")
print(f"  MSE:  {mse:.5f}")
print(f"  RMSE: {rmse:.5f}")
print(f"  MAE:  {mae:.5f}")


# 3. ЗАДАЧА КЛАССИФИКАЦИИ И ROC-КРИВАЯ
print("\n" + "=" * 50)
print("3. Задача классификации и ROC-кривая")
print("=" * 50)

# Воссоздаем целевую переменную 'room_type_Entire home/apt' так же, как во 2-й лабе
for df_item in [train_df, test_df]:
    if 'room_type_Entire home/apt' not in df_item.columns:
        df_item['room_type_Entire home/apt'] = (
            (df_item['room_type_Private room'] == 0) & 
            (df_item['room_type_Shared room'] == 0)
        ).astype(int)

target_cls_col = 'room_type_Entire home/apt'
room_type_cols = [col for col in train_df.columns if col.startswith('room_type_')]

X_train_cls = train_df.drop(columns=room_type_cols)
y_train_cls = train_df[target_cls_col]

X_test_cls = test_df.drop(columns=room_type_cols)
y_test_cls = test_df[target_cls_col]

# Инициализация и обучение DecisionTreeClassifier
dt_cls = DecisionTreeClassifier(max_depth=6, random_state=42)
dt_cls.fit(X_train_cls, y_train_cls)

y_pred_cls = dt_cls.predict(X_test_cls)
# Получаем вероятности для построения ROC-кривой (вероятности для класса 1)
y_proba_cls = dt_cls.predict_proba(X_test_cls)[:, 1]

acc = accuracy_score(y_test_cls, y_pred_cls)
print(f"Метрика Accuracy дерева классификации: {acc:.4f}")
print("\nОтчет классификации:")
print(classification_report(y_test_cls, y_pred_cls))

# Расчет ROC-кривой и AUC
fpr, tpr, thresholds = roc_curve(y_test_cls, y_proba_cls)
roc_auc = auc(fpr, tpr)
print(f"Значение ROC-AUC: {roc_auc:.4f}")

# Построение и сохранение ROC-кривой
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC-кривая (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Случайная модель')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR)', fontsize=12)
plt.title('ROC-кривая для Дерева Классификации', fontsize=14)
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png')
print("График ROC-кривой успешно сохранен как 'roc_curve.png'")
