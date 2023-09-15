import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# 示例数据
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([2, 3, 4, 4.5, 5, 7, 8, 8.5, 9, 10])
# 手动指定分段点
segment_points = [3, 7]
models = []  # 用于存储每个段的线性回归模型

for i in range(len(segment_points) + 1):
    if i == 0:
        X_segment = X[:segment_points[i]]
        y_segment = y[:segment_points[i]]
    elif i == len(segment_points):
        X_segment = X[segment_points[i - 1]:]
        y_segment = y[segment_points[i - 1]:]
    else:
        X_segment = X[segment_points[i - 1]:segment_points[i]]
        y_segment = y[segment_points[i - 1]:segment_points[i]]

    model = LinearRegression()
    model.fit(X_segment.reshape(-1, 1), y_segment)
    models.append(model)
    
#进行预测
y_pred = np.concatenate([model.predict(X_segment.reshape(-1, 1)) for model, X_segment in zip(models, X)])

#分段绘制线性回归结果
plt.scatter(X, y, label="Data")
plt.plot(X, y_pred, color='red', label="Piecewise Linear Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
