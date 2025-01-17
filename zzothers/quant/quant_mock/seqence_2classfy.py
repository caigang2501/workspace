import sys,os
sys.path.append(os.getcwd())
from zzothers.quant.data.tushare.data import data_ts1

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# data = data_ts1()
# data.to_csv('data/stock_data.csv')
data = pd.read_csv('data/stock_data.csv')

data['Price_Change'] = data['close'].pct_change()
data['Target'] = np.where(data['Price_Change'] > 0, 1, 0)
data = data.dropna()

features = ['open', 'high', 'low', 'close', 'vol']
X = data[features]
y = data['Target']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)

X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_lstm.shape[1], X_lstm.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))  # 二分类问题

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_lstm, y, epochs=10, batch_size=32)

y_pred = model.predict(X_lstm)

print(y_pred)
