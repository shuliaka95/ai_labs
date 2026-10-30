import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    roc_curve,
    auc
)

# ЧАСТЬ 1: MLP НА ДАТАСЕТЕ AIRBNB
print("=" * 60)
print("1. Загрузка данных Airbnb (из Лабы №1)")
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

X_train_cls = train_df.drop(columns=room_type_cols)
y_train_cls = train_df[target_cls_col]

X_test_cls = test_df.drop(columns=room_type_cols)
y_test_cls = test_df[target_cls_col]

X_train_reg = train_df.drop(columns=['price'])
y_train_reg = train_df['price']

X_test_reg = test_df.drop(columns=['price'])
y_test_reg = test_df['price']


# 1.1 Задача классификации (MLP vs Классические алгоритмы)
print("\n" + "=" * 60)
print("1.1 Обучение MLPClassifier и сравнение с другими алгоритмами")
print("=" * 60)

# Многослойный перцептрон (MLP)
mlp_cls = MLPClassifier(hidden_layer_sizes=(16, 8), activation='relu', max_iter=500, random_state=42)
mlp_cls.fit(X_train_cls, y_train_cls)

y_pred_mlp_cls = mlp_cls.predict(X_test_cls)
y_proba_mlp_cls = mlp_cls.predict_proba(X_test_cls)[:, 1]

acc_mlp = accuracy_score(y_test_cls, y_pred_mlp_cls)
print(f"MLP Classifier Accuracy: {acc_mlp:.4f}")

# Сравнение с логистической регрессией и случайным лесом
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_cls, y_train_cls)
acc_lr = accuracy_score(y_test_cls, log_reg.predict(X_test_cls))

rf_cls = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_cls.fit(X_train_cls, y_train_cls)
acc_rf = accuracy_score(y_test_cls, rf_cls.predict(X_test_cls))

print(f"Logistic Regression Accuracy: {acc_lr:.4f}")
print(f"Random Forest Accuracy:       {acc_rf:.4f}")

# Построение графика Loss Curve для MLP
plt.figure(figsize=(7, 4))
plt.plot(mlp_cls.loss_curve_, label='Training Loss (MLP)', color='purple')
plt.xlabel('Итерации (Эпохи)')
plt.ylabel('Функция потерь (Loss)')
plt.title('График обучения MLPClassifier (Airbnb Dataset)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('mlp_loss_curve.png')
print("График обучения MLP сохранен как 'mlp_loss_curve.png'")


# 1.2 Задача регрессии (MLPRegressor)
print("\n" + "=" * 60)
print("1.2 Задача регрессии (MLPRegressor)")
print("=" * 60)

mlp_reg = MLPRegressor(hidden_layer_sizes=(16, 8), activation='relu', max_iter=500, random_state=42)
mlp_reg.fit(X_train_reg, y_train_reg)

y_pred_mlp_reg = mlp_reg.predict(X_test_reg)
mse_mlp = mean_squared_error(y_test_reg, y_pred_mlp_reg)
mae_mlp = mean_absolute_error(y_test_reg, y_pred_mlp_reg)

print(f"MLP Regressor MSE: {mse_mlp:.5f}")
print(f"MLP Regressor MAE: {mae_mlp:.5f}")


# ЧАСТЬ 2: ЗАДАНИЕ «ВЫБОРЫ ПРЕЗИДЕНТА»
print("\n" + "=" * 60)
print("2. Выполнение спецзадания «Выборы президента»")
print("=" * 60)

# Генерация входных и выходных данных (если их нет на диске)
if not os.path.exists('dataIn.txt') or not os.path.exists('dataOut.txt'):
    np.random.seed(42)
    # 100 примеров, 12 бинарных ответов избирателей
    X_elections = np.random.randint(0, 2, size=(100, 12))
    # 100 примеров, 2 класса ([1, 0] - правящая партия, [0, 1] - оппозиция)
    Y_elections = np.zeros((100, 2), dtype=int)
    for i in range(100):
        if X_elections[i].sum() >= 6:
            Y_elections[i] = [1, 0]
        else:
            Y_elections[i] = [0, 1]

    np.savetxt('dataIn.txt', X_elections, fmt='%d')
    np.savetxt('dataOut.txt', Y_elections, fmt='%d')
    print("Созданы синтетические файлы dataIn.txt и dataOut.txt.")

# Чтение данных из файлов
X_election_data = np.loadtxt('dataIn.txt')
Y_election_data = np.loadtxt('dataOut.txt')

# Если матрица загружена в транспонированном виде (12 x N), приводим к (N x 12)
if X_election_data.shape[0] == 12:
    X_election_data = X_election_data.T
if Y_election_data.shape[0] == 2:
    Y_election_data = Y_election
