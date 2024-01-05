import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# 生成示例数据
np.random.seed(42)
data_size = 100
returns = np.random.normal(0, 1, data_size)
price = np.cumsum(returns)

# 构造特征
features = pd.DataFrame({'Returns': returns})

# 构造目标变量
target = price[1:]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# 构建支持向量机模型
svm_model = SVR()
svm_model.fit(X_train, y_train)

# 预测未来价格
svm_predictions = svm_model.predict(X_test)

# 可视化结果
plt.plot(y_test.values, label='True Price')
plt.plot(y_test.index, svm_predictions, label='SVM Predictions', color='orange')
plt.title('SVM Price Predictions')
plt.legend()
plt.show()
