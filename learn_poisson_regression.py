import numpy as np
import pandas as pd
import statsmodels.api as sm

# --- 1. 生成模拟数据 ---
# 假设我们有2个预测变量 (features):
# 'PeerRisk' (同伴风险, 0-10)
# 'ParentingStyle' (教养方式, 0=protective, 1=other)
# 我们要预测 'MonthlyIncidents' (每月纪律事件次数)

np.random.seed(42)
N = 500  # 500个样本

data = pd.DataFrame({
    'PeerRisk': np.random.uniform(0, 10, N),
    'ParentingStyle': np.random.randint(0, 2, N)
})

# 创建一个“真实”的线性关系 (但我们作为建模者是不知道的)
# log(E[Incidents]) = 0.1 + 0.2*PeerRisk + 0.5*ParentingStyle
true_log_mean = 0.1 + 0.2 * data['PeerRisk'] + 0.5 * data['ParentingStyle']

# 从泊松分布中生成结果
data['MonthlyIncidents'] = np.random.poisson(np.exp(true_log_mean))

print("--- 模拟数据 (前5行) ---")
print(data.head())
print("\n--- 数据描述 ---")
print(f"平均事件数: {data['MonthlyIncidents'].mean():.2f}")
print(f"事件数方差: {data['MonthlyIncidents'].var():.2f}")
print("(如果方差远大于均值，说明存在'过度离散', 泊松模型可能不合适)\n")

# --- 2. 拟合泊松回归模型 ---
# 准备 X (特征) 和 y (目标)
# 我们需要为截距 (intercept) 添加一个常数项
X = sm.add_constant(data[['PeerRisk', 'ParentingStyle']])
y = data['MonthlyIncidents']

# 拟合 GLM (广义线性模型), 指定 family 为 Poisson
poisson_model = sm.GLM(y, X, family=sm.families.Poisson())
poisson_results = poisson_model.fit()

# --- 3. 打印结果 ---
print("\n--- 泊松回归模型结果 ---")
print(poisson_results.summary())

print("\n--- 结论 ---")
print("模型摘要 (summary) 告诉我们 'PeerRisk' 和 'ParentingStyle' 是否显著影响 'MonthlyIncidents'。")
print("P>|z| 列（P值）越小越显著。")
print("这个模型是你 ZINB 和 NB 模型的最基础版本。")
