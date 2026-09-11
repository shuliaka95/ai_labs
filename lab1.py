import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Kaggle Dataset URL: https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data/data?select=AB_NYC_2019.csv

# Загрузка данных

print("*" * 50, "\n Import and display data \n", "*" * 50 )
df = pd.read_csv("AB_NYC_2019.csv.zip")
# head() - выводит первые пять строк info() - столбцы, типы и недоспустимые значения 
print("\n First 5 rows \n")
print(df.head())
print("\n Dataset info \n")
print(df.info())


# Поиск и заполнение пропущенных значений

print("\n Missing values check and handling \n")
print("\n Missing values before imputation \n") 
# isnull().sum() - считает количество пропущенных значений для каждого столбца
print(df.isnull().sum())
#Заполнение: мода - для категориальных, 0/медиана - для числовых 
df['name'] = df['name'].fillna(df['name'].mode()[0])
df['host_name'] = df['host_name'].fillna(df['host_name'].mode()[0])
df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
df['last_review'] = df['last_review'].fillna('1900-01-01')
print("\n Missing values after imputation \n")
print(df.isnull().sum())


# Преобразование категориальных данных

print("\n Categorical encoding (OHE) \n")
# Удаление уникальных идентификаторов и строковых данных высокой мощности
df_clean = df.drop(columns=['id', 'name', 'host_id', 'host_name', 'last_review', 'neighbourhood'])
# One-hot encoding (OHE) с drop_first=True для исключения мультиколлинеарности
df_encoded = pd.get_dummies(df_clean, columns=['neighbourhood_group', 'room_type'], drop_first=True)
print("\n Encoded columns \n")
print(df_encoded.columns.tolist())


# Разбиение на выборки и нормализации

print( "\n Train-Test Split & Normalization \n")
# train_test_split разделяет данные на обучающую и тестовую выборки
train_df, test_df = train_test_split(df_encoded, test_size=0.3, random_state=42)
# MinMaxScaler приводить числовые значения к диапазону [0, 1]
scaler = MinMaxScaler()
num_cols = ['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']
train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
test_df[num_cols] = scaler.transform(test_df[num_cols])
print("\n Processed Train Head \n")
print(train_df.head())


# Сохранение обработанных данных

print("\n Saving processed datasets \n")
train_df.to_csv("processed_train.csv", index=False)
test_df.to_csv("processed_test.csv", index=False)
print("\n Processing complete. Files saved successfully. \n")
