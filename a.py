import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 1. 读取数据
data = pd.read_excel('data.xlsx', parse_dates=['日期'])
data.set_index('日期', inplace=True)
assets = ['黄金ETF', '红利低波ETF', '纳斯达克ETF']
prices = data[assets]
returns_full = prices.pct_change().dropna()
# 2. 初始配置
weights = np.array([0.3, 0.4, 0.3])   # 初始权重
init_capital = 1_000_000              # 初始资金（元）
# 3. VaR 样本窗口：2023-12-29 含当天，往前 500 日
hist_end = '2023-12-29'
hist_returns = returns_full.loc[:hist_end].iloc[-500:]
port_r = hist_returns.dot(weights)   # 组合回报
n = len(port_r)
alpha = 0.01                         # 99% VaR
# 计算当日组合市值 V_n 及份额
price_n = prices.loc[hist_end]
shares = weights * init_capital / price_n
V_n = shares.dot(price_n)

# --- 4.1 传统历史模拟 VaR ---
loss_hist = -V_n * port_r.values
df_hist = pd.DataFrame({
    'loss': loss_hist,
    'weight': np.ones(n)/n
}).sort_values('loss', ascending=False)
cumw = df_hist['weight'].cumsum()
VaR_hist = df_hist.loc[cumw >= alpha, 'loss'].iloc[0]
df_hist.to_csv('historical_loss_weights.csv', index=False)
print(f"传统历史模拟 VaR(99%): {VaR_hist:.2f} 元")

# --- 4.2 时间加权历史模拟 VaR (λ=0.99) ---
lam = 0.99
exps = np.arange(n-1, -1, -1)
w_tw = lam**exps * (1-lam) / (1-lam**n)
w_tw /= w_tw.sum()
df_tw = pd.DataFrame({
    'loss': loss_hist,
    'weight': w_tw
}).sort_values('loss', ascending=False)
cumw2 = df_tw['weight'].cumsum()
VaR_tw = df_tw.loc[cumw2 >= alpha, 'loss'].iloc[0]
df_tw.to_csv('time_weighted_loss_weights.csv', index=False)
print(f"时间加权历史模拟 VaR(99%, λ=0.99): {VaR_tw:.2f} 元")

# --- 4.3 EWMA 波动率加权 VaR (λ=0.95) ---
lam2 = 0.95
s2 = port_r.var()
sigma = []
for r in port_r:
    s2 = lam2*s2 + (1-lam2)*r*r
    sigma.append(math.sqrt(s2))
sigma = np.array(sigma)
sigma_n1 = sigma[-1]
adj_r = port_r.values * (sigma_n1/sigma)
loss_ewma = -V_n * adj_r
w_ewma = lam2**exps * (1-lam2) / (1-lam2**n)
w_ewma /= w_ewma.sum()
df_ewma = pd.DataFrame({
    'loss': loss_ewma,
    'weight': w_ewma
}).sort_values('loss', ascending=False)
cumw3 = df_ewma['weight'].cumsum()
VaR_ewma = df_ewma.loc[cumw3 >= alpha, 'loss'].iloc[0]
df_ewma.to_csv('ewma_loss_weights.csv', index=False)
print(f"EWMA 波动率加权 VaR(99%, λ=0.95): {VaR_ewma:.2f} 元")

# --- 5. 静态组合回测 2024-01-01 至 2025-04-18 ---
start, end = '2024-01-01', '2025-04-18'
prices_period = prices.loc[start:end]
dates = prices_period.index
static_values = prices_period.dot(shares)
pd.DataFrame({
    'date': dates,
    'portfolio_value': static_values.values
}).to_csv('portfolio_value.csv', index=False)
plt.figure()
plt.plot(dates, static_values, label='静态组合')
plt.xlabel('日期'); plt.ylabel('组合市值 (元)')
plt.title(f'静态组合市值 {start} 至 {end}')
plt.tight_layout()
plt.savefig('static_portfolio_value.png')
plt.close()

# --- 6. Kupiec 非违约率检验 ---
def kupiec_test(returns, VaR, alpha):
    losses = -V_n * returns
    x = (losses > VaR).sum()
    n = len(losses)
    p = alpha
    L0 = (1-p)**(n-x) * p**x
    L1 = (1-x/n)**(n-x) * (x/n)**x if x>0 and x<n else 1e-10
    lr = -2*(math.log(L0) - math.log(L1))
    return lr, 1-chi2.cdf(lr, df=1)
rets = static_values.pct_change().dropna()
for name, va in [('传统', VaR_hist), ('时间加权', VaR_tw), ('EWMA', VaR_ewma)]:
    lr, pv = kupiec_test(rets, va, alpha)
    print(f"{name} Kupiec LR: {lr:.2f}, p-value: {pv:.4f}")

# --- 7. 分析二：每 5 交易日网格搜索动态重平衡 VaR 最小化 ---
grid_points = 500   # 搜索精度
weight_range = np.linspace(-2, 2, grid_points)
rebalance_dates = dates[::5]  # 更新间隔为5天
port_dyn = pd.Series(index=dates, dtype=float)
current_shares = shares.copy()
for d in rebalance_dates:
    # 1) 计算旧市值
    V_old = prices.loc[d].dot(current_shares)
    # 2) 网格搜索最小 VaR 权重
    hist = returns_full.loc[:d].iloc[-500:]
    best_var, best_w = np.inf, None
    for w1 in weight_range:
        for w2 in weight_range:
            w3 = 1 - w1 - w2
            if -2 <= w3 <= 2:
                w = np.array([w1, w2, w3])
                var_i = -np.quantile(hist.dot(w), alpha)
                if var_i < best_var:
                    best_var, best_w = var_i, w
    # 3) 打印并重分配份额（市值不变）
    print(
        f"{d.strftime('%Y-%m-%d')} 重平衡最小VaR权重："
        + ", ".join(f"{asset}:{w:.4f}" for asset, w in zip(assets, best_w))
    )
    current_shares = best_w * V_old / prices.loc[d]
    port_dyn.loc[d] = V_old
    # 4) 填充下一个重平衡日前的市值
    next_idx = np.where(dates==d)[0][0] + 1
    next_d = rebalance_dates[list(rebalance_dates).index(d)+1] \
             if d != rebalance_dates[-1] else None
    if next_d is not None:
        for t in dates[next_idx:dates.get_loc(next_d)]:
            port_dyn.loc[t] = prices.loc[t].dot(current_shares)
# 最后填充剩余
for t in dates:
    if pd.isna(port_dyn.loc[t]):
        port_dyn.loc[t] = prices.loc[t].dot(current_shares)
# 保存 & 绘图对比
pd.DataFrame({
    'date': dates,
    'dynamic_value': port_dyn.values
}).to_csv('dynamic_portfolio_value.csv', index=False)
plt.figure()
plt.plot(dates, static_values, label='静态组合')
plt.plot(dates, port_dyn,    label='动态(5日网格VaR)组合')
plt.xlabel('日期'); plt.ylabel('组合市值 (元)')
plt.title(f'静态 vs 动态组合 {start} 至 {end}')
plt.legend(); plt.tight_layout()
plt.savefig('static_vs_dynamic_grid.png')
plt.close()
