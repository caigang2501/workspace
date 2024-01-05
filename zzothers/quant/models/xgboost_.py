import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 生成示例数据
data = {'Feature1': [1, 2, 3, 4, 5], 'Feature2': [5, 4, 3, 2, 1], 'Price': [10, 20, 30, 40, 50]}
df = pd.DataFrame(data)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(df[['Feature1', 'Feature2']], df['Price'], test_size=0.2, random_state=42)

# 构建 XGBoost 模型
model = XGBRegressor()
model.fit(X_train, y_train)

# 预测未来数据点
predictions = model.predict(X_test)
print(predictions)
