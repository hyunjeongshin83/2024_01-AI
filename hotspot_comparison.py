import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from prophet import Prophet

# Load the data into a DataFrame
data = pd.read_csv('wireshark_1.csv')

# Display the first few rows of the dataframe to understand its structure
dataF = pd.DataFrame(data)
print("Initial Data:")
print(dataF.head())

# 이상치 확인
print("Data Description:")
print(dataF.describe())

# 범주형 데이터 인코딩
label_encoder = LabelEncoder()
dataF['Source_Encoded'] = label_encoder.fit_transform(dataF['Source'])
dataF['Destination_Encoded'] = label_encoder.fit_transform(dataF['Destination'])
dataF['Protocol_Encoded'] = label_encoder.fit_transform(dataF['Protocol'])

# 정규화
scaler = StandardScaler()
dataF[['Time', 'Length']] = scaler.fit_transform(dataF[['Time', 'Length']])

# Source 빈도 계산
source_counts = dataF['Source'].value_counts()
print("Source counts:\n", source_counts)
dataF['source_counts'] = dataF['Source'].map(source_counts)

# 임계값 설정 (예: 5)
threshold = 5

# 임계값을 넘어가는 Source 값들 출력
high_frequency_sources = source_counts[source_counts >= threshold]
print(f"Sources with counts >= {threshold}:\n", high_frequency_sources)

# 핫스팟(1)과 일반 트래픽(0)으로 레이블링합니다. Source 빈도가 임계값 이상이면 1로 설정
dataF['traffic_type'] = 0
dataF.loc[dataF['Source'].isin(high_frequency_sources.index), 'traffic_type'] = 1
print("Data with traffic_type labeled:")
print(dataF.head(10))

# 최종 데이터
processed_data = dataF[['Time', 'Source_Encoded', 'Destination_Encoded', 'Protocol_Encoded', 'Length', 'traffic_type']]

# Split the data into training and testing sets
X = processed_data.drop('traffic_type', axis=1)
y = processed_data['traffic_type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 정의 및 훈련
gru_model = Sequential()
gru_model.add(layers.GRU(64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
gru_model.add(layers.GRU(32))
gru_model.add(layers.Dense(1, activation='sigmoid'))
gru_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

X_train_gru = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_gru = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))
gru_model.fit(X_train_gru, y_train, epochs=10, batch_size=32)

prophet_data = pd.DataFrame({
    'ds': pd.to_datetime(dataF.index, unit='s'),
    'y': dataF['traffic_type']
})
prophet_model = Prophet()
prophet_model.fit(prophet_data)

# Prophet 예측 함수
def prophet_predict(X):
    future = pd.DataFrame({'ds': pd.to_datetime(X.index, unit='s')})  # 인덱스 형식 변환
    forecast = prophet_model.predict(future)
    future = future.merge(forecast[['ds', 'yhat']], on='ds')
    return future['yhat']

# 모델 평가 함수
def evaluate_model(model, X_test, y_test, name, is_prophet=False, is_stacking=False):
    if is_prophet:
        predictions = model
        final_predictions = (predictions >= 0.5).astype(int)
    elif is_stacking:
        predictions = model
        final_predictions = (predictions >= 0.5).astype(int)
    else:
        predictions = model.predict(X_test_gru)
        final_predictions = (predictions > 0.5).astype(int)

    accuracy = accuracy_score(y_test, final_predictions)
    precision = precision_score(y_test, final_predictions)
    recall = recall_score(y_test, final_predictions)
    f1 = f1_score(y_test, final_predictions)
    auc = roc_auc_score(y_test, final_predictions)
    
    print(f"{name} 모델 성능:")
    print(f"정확도: {accuracy:.4f}")
    print(f"재현율: {recall:.4f}")
    print(f"정밀도: {precision:.4f}")
    print(f"F1 점수: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    
    # 예측 값 출력
    print(f"{name} Predictions: {predictions[:10]}")
    print(f"{name} Final Predictions: {final_predictions[:10]}")
    
    return accuracy, precision, recall, f1, auc

# GRU 모델 평가
gru_scores = evaluate_model(gru_model, X_test_gru, y_test, "GRU")

# Prophet 모델 평가
prophet_predictions = prophet_predict(X_test)
prophet_scores = evaluate_model(prophet_predictions, X_test, y_test, "Prophet", is_prophet=True)

# Stacking 모델 평가
def stacking_model(X):
    X_gru = X.values.reshape((X.shape[0], X.shape[1], 1))
    gru_predictions = gru_model.predict(X_gru).flatten()
    prophet_predictions = prophet_predict(X)
    combined_predictions = 0.7 * gru_predictions + 0.3 * prophet_predictions
    return combined_predictions

stacking_predictions = stacking_model(X_test)
stacking_scores = evaluate_model(stacking_predictions, X_test, y_test, "Stacking", is_stacking=True)



# 성능 비교 그래프
models = ["GRU", "Prophet", "Stacking"]
metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
scores = {
    "GRU": gru_scores,
    "Prophet": prophet_scores,
    "Stacking": stacking_scores
}

plt.figure(figsize=(10, 6))
for i, metric in enumerate(metrics):
    plt.subplot(2, 3, i+1)
    metric_scores = [scores[model][i] for model in models]
    max_score = max(metric_scores)
    bars = plt.bar(models, metric_scores)
    plt.ylim(0, 1)
    plt.title(metric)
    for bar, score in zip(bars, metric_scores):
        height = bar.get_height()
        plt.annotate(f"{score:.4f}",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom')
        if score == max_score:
            bar.set_color('r')
plt.tight_layout()
plt.show()
