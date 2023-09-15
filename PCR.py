import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# 生成示例数据
np.random.seed(0)
X = np.random.rand(100, 5)  # 100个样本，5个特征
y = 2*X[:, 0] + 3*X[:, 1] + 0.5*X[:, 2] + np.random.rand(100)
# 生成示例数据
np.random.seed(0)
X = np.random.rand(100, 5)  # 100个样本，5个特征
y = 2*X[:, 0] + 3*X[:, 1] + 0.5*X[:, 2] + np.random.rand(100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 指定主成分数量，这里选择2个主成分
n_components = 2
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
# 创建线性回归模型
model = LinearRegression()
model.fit(X_train_pca, y_train)
y_pred = model.predict(X_test_pca)

# 计算模型性能，例如均方根误差（RMSE）
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse}")
