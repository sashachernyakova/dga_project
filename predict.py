import pandas as pd
import joblib
from generate_features import generate_features

# Загрузка данных
test_df = pd.read_csv('test.csv')

# Генерация признаков
test_features = generate_features(test_df.copy())

X_test = test_features.drop(columns=['domain'])

# Загрузка модели
model = joblib.load('dga_model.pkl')

# Предсказание
test_df['is_dga'] = model.predict(X_test)

# Сохранение предсказаний
# первую колонку с индексом не добавляем
test_df[['domain', 'is_dga']].to_csv('prediction.csv', index=False)
