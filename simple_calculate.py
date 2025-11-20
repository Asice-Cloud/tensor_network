import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

v1 = np.array([1517,1577,1629,1670,1818,1891])
v11 = np.array([1011,1046,1082,1112,1197,1263])
v2 = np.array([219.9,247.9,262.8,283,319.6,352.1])
v22 = np.array([27.15,46.93,61.9,73.4,116.2,144.6])
I = np.array([300,320,334,351,388,416])/1000

v87 = (v1+v2)/2
v85 = (v11+v22)/2

print(v87)
print(v85)

B_horizon = np.zeros(len(v1))
for i in range(len(v1)):
    B_horizon[i] = 16*math.pi*250*I[i]/((5**(1.5))*0.2424)*(10**(-7))

print(B_horizon)

# 线性拟合与可视化
def linear_func(B, a, b):
    return a * B + b

# 拟合 v87
params_87, _ = curve_fit(linear_func, B_horizon, v87)
# 拟合 v85
params_85, _ = curve_fit(linear_func, B_horizon, v85)

print("v87拟合: a=%.3f, b=%.3f" % tuple(params_87))
print("v85拟合: a=%.3f, b=%.3f" % tuple(params_85))

# 可视化
plt.scatter(B_horizon, v87, label='v87 data')
plt.plot(B_horizon, linear_func(B_horizon, *params_87), label='v87 fit')
plt.scatter(B_horizon, v85, label='v85 data')
plt.plot(B_horizon, linear_func(B_horizon, *params_85), label='v85 fit')
plt.xlabel('B_horizon')
plt.ylabel('v')
plt.legend()
plt.title('Rb87 & Rb85 vs B_horizon')
plt.savefig('Rb87_Rb85_vs_B_horizon_fit.png')

# 计算相对误差
v87_fit = linear_func(B_horizon, *params_87)
v85_fit = linear_func(B_horizon, *params_85)

rel_err_v87 = np.abs((v87 - v87_fit) / v87)
rel_err_v85 = np.abs((v85 - v85_fit) / v85)

print('v87相对误差:', rel_err_v87)
print('v85相对误差:', rel_err_v85)
print('v87相对误差均值: %.4f' % rel_err_v87.mean())
print('v85相对误差均值: %.4f' % rel_err_v85.mean())

# 计算朗德g因子
mu_B = 9.274009994e-24  # 玻尔磁子 (J/T)
h = 6.62607015e-34      # 普朗克常数 (J·s)

gF_87 = params_87[0] * h / mu_B
gF_85 = params_85[0] * h / mu_B

print('Rb87 朗德g因子: %.5f' % gF_87)
print('Rb85 朗德g因子: %.5f' % gF_85)

# 计算地磁场水平分量 B_earth
# 假设已知某一组频率 v 和外加直流磁场 B_dc
# 这里以 v87 拟合为例，B_dc 可自定义或用 I 数组

# 例如：假设外加直流磁场为 I（单位 T），用第一个点演示
B_dc = I[0]  # 单位 T
v_meas = v87[0]  # 观测频率
A = params_87[0]  # 线性拟合斜率


# 如果需要批量计算所有点的地磁场分量：
B_earth_all = (v87 - A * I) / A
print('所有点地磁场水平分量 B_earth:', B_earth_all)

# 计算所有点地磁场水平分量的平均值
B_earth_mean = B_earth_all.mean()
print('地磁场水平分量平均值 B_earth_mean: %.6e T' % B_earth_mean)
