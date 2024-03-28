from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# 生成示例数据
data = {'Feature1': [1, 2, 3, 4, 5], 'Feature2': [5, 4, 3, 2, 1], 'Price': [10, 20, 30, 40, 50]}
df = pd.DataFrame(data)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(df[['Feature1', 'Feature2']], df['Price'], test_size=0.2, random_state=42)

# 构建各个基模型
model1 = LinearRegression()
model2 = RandomForestRegressor()
model3 = SVR()

# 构建投票集成模型
ensemble_model = VotingRegressor(estimators=[('lr', model1), ('rf', model2), ('svm', model3)])

# 训练集成模型
ensemble_model.fit(X_train, y_train)

# 预测未来数据点
predictions = ensemble_model.predict(X_test)
print(predictions)
