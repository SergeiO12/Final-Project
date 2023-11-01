import pickle
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# Обученная модель
# Лист с порядком признаков
# Признаки для тестового датасета

MODEL_FILE = '_project_model.pkl'
FEATURE_ORDER_FILE = '_feature_order.pkl'
TEST_DATA_FILE = '_test_data.pkl'


# Загружаем сериализованные объекты с помощью кода pickle

with open(MODEL_FILE, 'rb') as inp:
    LR = pickle.load(inp)

with open(TEST_DATA_FILE, 'rb') as inp:
    X_test = pickle.load(inp)

with open(FEATURE_ORDER_FILE, 'rb') as inp:
    feature_order = pickle.load(inp)

# Создадим функцию способную выводить стоимость жилья при указании года постройки строительного объекта
# Функция будет возвращать стоимость жилья по адресу http://localhost:8000/predict?target=
# В методе предусмотрена обработка ошибок - при указании неверных данных будет выводится слово 'ERROR' в предсказании

@app.route('/predict', methods=['GET'])
def predict():
    target = request.args.get('Year_built')
    try:
        target = int(target)
        test_features = X_test[X_test['Year_built'] == target][feature_order].values
        res = LR.predict(test_features)[0]
        #print(res)
        return jsonify({'prediction': round(res, 4), 'Year_built': str(target), 'response_status': 'OK'})
    except:
        return jsonify({'prediction': -1, 'Year_built': str(target), 'response_status': 'ERROR'})


if __name__ == '__main__':
    app.run('0.0.0.0', port=8000, debug=True)
