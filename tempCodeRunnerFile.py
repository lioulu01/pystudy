import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt

# 生成示例数据
np.random.seed(0)
X = np.linspace(0, 10, 100)
true_slope = 2
true_intercept = 1
y = true_slope * X + true_intercept + np.random.normal(0, 1, 100)

# 定义贝叶斯线性回归模型
with pm.Model() as model:
    # 定义先验分布
    slope = pm.Normal("slope", mu=0, sigma=10)
    intercept = pm.Normal("intercept", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=1)
    
    # 定义线性模型
    mu = slope * X + intercept
    
    # 定义似然函数
    likelihood = pm.Normal("y", mu=mu, sigma=sigma, observed=y)
    
    # 运行贝叶斯推断
    trace = pm.sample(2000, tune=1000, chains=2)

# 绘制后验分布
pm.traceplot(trace)
plt.show()

# 获取后验分布的统计信息
posterior_summary = pm.summary(trace)
print(posterior_summary)

# 绘制后验分布的预测结果
plt.scatter(X, y, label="Data")
plt.xlabel("X")
plt.ylabel("y")

for i in range(100):
    slope_sample = trace["slope"][i]
    intercept_sample = trace["intercept"][i]
    y_pred = slope_sample * X + intercept_sample
    plt.plot(X, y_pred, color="gray", alpha=0.1)

plt.plot(X, true_slope * X + true_intercept, color="red", label="True Regression Line")
plt.legend()
plt.show()
import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt

# 生成示例数据
np.random.seed(0)
X = np.linspace(0, 10, 100)
true_slope = 2
true_intercept = 1
y = true_slope * X + true_intercept + np.random.normal(0, 1, 100)

# 定义贝叶斯线性回归模型
with pm.Model() as model:
    # 定义先验分布
    slope = pm.Normal("slope", mu=0, sigma=10)
    intercept = pm.Normal("intercept", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=1)
    
    # 定义线性模型
    mu = slope * X + intercept
    
    # 定义似然函数
    likelihood = pm.Normal("y", mu=mu, sigma=sigma, observed=y)
    
    # 运行贝叶斯推断
    trace = pm.sample(2000, tune=1000, chains=2)

# 绘制后验分布
pm.traceplot(trace)
plt.show()

# 获取后验分布的统计信息
posterior_summary = pm.summary(trace)
print(posterior_summary)

# 绘制后验分布的预测结果
plt.scatter(X, y, label="Data")
plt.xlabel("X")
plt.ylabel("y")

for i in range(100):
    slope_sample = trace["slope"][i]
    intercept_sample = trace["intercept"][i]
    y_pred = slope_sample * X + intercept_sample
    plt.plot(X, y_pred, color="gray", alpha=0.1)

plt.plot(X, true_slope * X + true_intercept, color="red", label="True Regression Line")
plt.legend()
plt.show()
