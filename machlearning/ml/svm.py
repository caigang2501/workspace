import numpy as np

# 生成随机数据集
np.random.seed(0)
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
y = [-1] * 20 + [1] * 20

def hinge_loss(X, y, w, b, C):
    loss = 1 - y * (np.dot(X, w) + b)
    loss[loss < 0] = 0
    return 0.5 * np.dot(w, w) + C * np.sum(loss)

def gradient(X, y, w, b, C):
    y_pred = np.dot(X, w) + b
    margin = 1 - y * y_pred
    misclassified = margin > 0
    dw = w - C * np.dot(X.T, misclassified * y)
    db = -C * np.sum(misclassified * y)
    return dw, db

def svm_train(X, y, learning_rate=0.01, C=1.0, epochs=1000):
    # 初始化权重和偏差
    w = np.zeros(X.shape[1])
    b = 0
    
    # 梯度下降训练模型
    for epoch in range(epochs):
        dw, db = gradient(X, y, w, b, C)
        w -= learning_rate * dw
        b -= learning_rate * db
        
        # 打印损失
        if epoch % 100 == 0:
            loss = hinge_loss(X, y, w, b, C)
            print(f'Epoch {epoch}, Loss: {loss}')
    
    return w, b

# 添加偏置列
X_with_bias = np.c_[X, np.ones(X.shape[0])]

# 训练SVM模型
w, b = svm_train(X_with_bias, y)

# 输出训练得到的权重和偏差
print('Weights:', w)
print('Bias:', b)
