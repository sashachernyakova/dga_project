import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import joblib
from generate_features import generate_features

# тестирующая выборка
df_valid = pd.read_csv('val.csv')

# генерация признаков
df_val = generate_features(df_valid.copy())

# удаляем столбец domain, так как из него уже извлекли все остальные числовые признаки
X_val = df_val.drop(["is_dga", "domain"], axis=1)
y_val = df_val["is_dga"].astype(int)

# загрузка модели
model = joblib.load('dga_model.pkl')

# predict_proba возвращает массив вероятностей. 
# Для бинарной классификации массив имеет форму (n_samples, 2), где n_samples — 
# количество образцов, а 2 — количество классов. Каждая строка содержит две
# вероятности: первая — для класса 0, вторая — для класса 1.
# Срез выбирает все строки (:) и только второй столбец (1).
# y_pred_proba будет одномерным массивом, содержащим вероятности принадлежности
# каждого образца к классу 1.
y_pred_proba = model.predict_proba(X_val)[:, 1]

y_pred = model.predict(X_val)

# оценка, расчет метрик
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
# ravel() преобразует многомерный массив в одномерный
tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
auc = roc_auc_score(y_val, y_pred_proba)

# Запись результатов
with open('validation.txt', 'w') as f:
    f.write(f"True positive: {tp}\n")
    f.write(f"False positive: {fp}\n")
    f.write(f"False negative: {fn}\n")
    f.write(f"True negative: {tn}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1: {f1:.4f}\n")
    f.write(f"AUC: {auc:.4f}\n")
