import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# выгрузим очищенный датасет
estate_data = pd.read_csv('final_dataset.csv')

# Создадим матрицу наблюдений X и вектор правильных ответов Y
Y=estate_data['Year_built']
X=estate_data.drop('Year_built', axis=1)

# Разделим выборку на обучающую и тестовую выборки.
X_train, X_test, y_train, y_test  = train_test_split(X, Y, test_size = 0.3, shuffle = True)
estate_data['Year_built'] = list(range(X.shape[0]))
# 1. Создаем модель
# 2. Обучаем модель
# 3. Предсказываем значения для выборки
# 4. Используем метрики MAE, MSE, R2_score

LR = LinearRegression()
LR.fit(X_train.drop(columns=['Year_built']), y_train)
Y_LR = LR.predict(X_test.drop(columns=['Year_built']))
print ('MAE:', round (mean_absolute_error(y_test, Y_LR),3))
print ('MSE:', round (mean_squared_error(y_test, Y_LR)**(1/2),3))
print ('R2_score:', round (r2_score(y_test, Y_LR),3))


# Сериализуем обученную модель с помощью pickle в файл "_project_model.pkl"
MODEL_FILE = '_project_model.pkl'
FEATURE_ORDER_FILE = '_feature_order.pkl'
TEST_DATA_FILE = '_test_data.pkl'


# Код сериализации моделей в MODEL_FILE с помощью pickle
with open(MODEL_FILE, 'wb') as output:
    pickle.dump(LR, output)

# Код сериализации тестовых данных в TEST_DATA_FILE с помощью pickle
# Формат сохранения тестовых данных - Pandas DataFrame
with open(TEST_DATA_FILE, 'wb') as output:
    pickle.dump(X_test, output)

# Сохраним в отдельный файл feature_order - порядок названий признаков в нашем датасете
feature_order = estate_data.columns.tolist()
feature_order.remove('Year_built')
print('Feature order:', feature_order)

# Код сериализации feature_order в  FEATURE_ORDER_FILE с помощью pickle
with open(FEATURE_ORDER_FILE, 'wb') as output:
    pickle.dump(feature_order, output)

