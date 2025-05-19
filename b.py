import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 1. 读取数据
data = pd.read_excel('data.xlsx', parse_dates=['日期'])
data.set_index('日期', inplace=True)
assets = ['黄金ETF', '红利低波ETF', '纳斯达克ETF']
prices = data[assets]
returns_full = prices.pct_change().dropna()
# 选择 2023-12-29 之前的 500 个交易日的数据
hist_end = '2023-12-29'
hist_returns = returns_full.loc[:hist_end].iloc[-500:]  # 获取2023-12-29之前的500个交易日的数据
# 2. 波动性分析：计算滚动窗口波动率（30日窗口）
window = 30
volatility = hist_returns.rolling(window).std() * np.sqrt(window)
plt.figure(figsize=(10, 6))
plt.plot(volatility['黄金ETF'], label='黄金ETF 波动率', color='blue')
plt.plot(volatility['红利低波ETF'], label='红利低波ETF 波动率', color='green')
plt.plot(volatility['纳斯达克ETF'], label='纳斯达克ETF 波动率', color='red')
plt.title('滚动30日波动率')
plt.xlabel('日期')
plt.ylabel('波动率')
plt.legend()
plt.tight_layout()
plt.savefig('滚动30日波动率.png')  # 保存图像
plt.close()

