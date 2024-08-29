import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from generate_features import generate_features

# обучающая выборка
df = pd.read_csv('train.csv')

# генерация признаков
df_train = generate_features(df.copy())

# Удаление строк, где значение в столбце 'len' меньше 5, так как это больше шумы
df_train = df_train[df_train['len'] >= 5]

# Удаляем столбец domain, так как из него уже извлекли все остальные числовые признаки
X_train = df_train .drop(["is_dga", "domain"], axis=1)
y_train = df_train ["is_dga"].astype(int)

# обучение модели (Случайный лес)
model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

# Сохранение модели
joblib.dump(model, 'dga_model.pkl')
