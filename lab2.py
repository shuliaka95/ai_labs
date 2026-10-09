import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

# Загрузка и подготовка данных
print("*" * 50, "\n Загрузка обработанных датасетов\n", "*" * 50)

train_df = pd.read_csv("processed_train.csv")
test_df = pd.read_csv("processed_test.csv")

print(f"Размер обучающей выборки: {train_df.shape}")
print(f"Размер тестовой выборки: {test_df.shape}")


# Задача регресии (прогноз 'price')
print("\n" + "*" * 50, "\n Задача регрессии (прогноз 'price')\n", "*" * 50)

X_train_reg = train_df.drop(columns=['price'])
y_train_reg = train_df['price']

X_test_reg = test_df.drop(columns=['price'])
y_test_reg = test_df['price']

# Линейная регрессия
lin_reg = LinearRegression()
lin_reg.fit(X_train_reg, y_train_reg)

y_pred_reg = lin_reg.predict(X_test_reg)

mse = mean_squared_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_reg, y_pred_reg)

print("Метрики Линейной Регрессии:")
print(f"  MSE:  {mse:.5f}")
print(f"  RMSE: {rmse:.5f}")
print(f"  MAE:  {mae:.5f}")

# Регуляризация Ridge (L2)
ridge_reg = Ridge(alpha=1.0)
ridge_reg.fit(X_train_reg, y_train_reg)
y_pred_ridge = ridge_reg.predict(X_test_reg)

mae_ridge = mean_absolute_error(y_test_reg, y_pred_ridge)
print(f"  MAE (Ridge L2): {mae_ridge:.5f}")


# 3.Задача класификации
print("\n" + "*" * 50, "\n Задача классификации\n", "*" * 50)

# Добавляем столбец 'room_type_Entire home/apt'
# (Он равен 1, если и Private room = 0, и Shared room = 0)
train_df['room_type_Entire home/apt'] = (
    (train_df['room_type_Private room'] == 0) & (train_df['room_type_Shared room'] == 0)
).astype(int)

test_df['room_type_Entire home/apt'] = (
    (test_df['room_type_Private room'] == 0) & (test_df['room_type_Shared room'] == 0)
).astype(int)

target_cls_col = 'room_type_Entire home/apt'

# Исключаем из X все признаки, связанные с room_type, чтобы избежать утечки данных (data leakage)
room_type_cols = [col for col in train_df.columns if col.startswith('room_type_')]

X_train_cls = train_df.drop(columns=room_type_cols)
y_train_cls = train_df[target_cls_col]

X_test_cls = test_df.drop(columns=room_type_cols)
y_test_cls = test_df[target_cls_col]

# Логистическая регрессия
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_cls, y_train_cls)

y_pred_cls = log_reg.predict(X_test_cls)

acc = accuracy_score(y_test_cls, y_pred_cls)
prec = precision_score(y_test_cls, y_pred_cls)
rec = recall_score(y_test_cls, y_pred_cls)
f1 = f1_score(y_test_cls, y_pred_cls)

print("Метрики Логистической Регрессии:")
print(f"  Accuracy:  {acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall:    {rec:.4f}")
print(f"  F1-Score:  {f1:.4f}")

print("\nПодробный отчет классификации (Classification Report):")
print(classification_report(y_test_cls, y_pred_cls))

# Построение и сохранение матрицы ошибок
cm = confusion_matrix(y_test_cls, y_pred_cls)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Матрица ошибок (Confusion Matrix)')
plt.ylabel('Истинные классы')
plt.xlabel('Предсказанные классы')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("\nГрафик матрицы ошибок успешно сохранен как 'confusion_matrix.png'")
