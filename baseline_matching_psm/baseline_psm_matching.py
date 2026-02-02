# -*- coding: utf-8 -*-
"""
基期匹配PSM分析
====================
核心思路：
1. 选择2009年作为基期（policy实施前一年）
2. 只在2009年进行PSM匹配（1:2近邻匹配，有放回，卡尺=0.05）
3. 记录匹配成功的城市名单（处理组+对照组）
4. 用这份名单从原始数据中提取这些城市2007-2019年的所有数据
5. 后续可以进行DID分析
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import rcParams
from collections import Counter

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("基期匹配PSM分析 (Baseline Year Matching)")
print("=" * 80)

# ==================== 参数设置 ====================
BASELINE_YEAR = 2009  # 基期年份（policy实施前一年）
RATIO = 2  # 1:2匹配
CALIPER = 0.05  # 卡尺限制

# 定义匹配变量（协变量）
covariates = ['ln_pgdp', 'ln_pop_density', 'industrial_advanced',
              'ln_fdi_openness', 'financial_development']

print(f"\n参数设置:")
print(f"  基期年份: {BASELINE_YEAR}")
print(f"  匹配比例: 1:{RATIO}")
print(f"  卡尺限制: {CALIPER}")
print(f"\n匹配变量（协变量）:")
for i, var in enumerate(covariates, 1):
    print(f"  {i}. {var}")

# ==================== 步骤1: 读取数据 ====================
print("\n" + "=" * 80)
print("步骤1: 读取数据")
print("=" * 80)

# 读取原始数据
df = pd.read_excel('../CEADs_最终数据集_2007-2019_V2_插值版_带DID_treat_缩尾版.xlsx')

print(f"\n原始数据概况:")
print(f"  总样本数: {len(df)}")
print(f"  城市数量: {df['city_name'].nunique()}")
print(f"  年份范围: {df['year'].min()} - {df['year'].max()}")
print(f"  列名: {list(df.columns)}")

# ==================== 步骤2: 提取基期数据 ====================
print("\n" + "=" * 80)
print(f"步骤2: 提取基期 ({BASELINE_YEAR}) 数据")
print("=" * 80)

baseline_data = df[df['year'] == BASELINE_YEAR].copy()

print(f"\n基期数据概况:")
print(f"  基期样本数: {len(baseline_data)}")
print(f"  基期城市数: {baseline_data['city_name'].nunique()}")

# 分离处理组和对照组
treated_baseline = baseline_data[baseline_data['treat'] == 1].copy()
control_baseline = baseline_data[baseline_data['treat'] == 0].copy()

n_treated = len(treated_baseline)
n_control = len(control_baseline)

print(f"\n处理组样本数: {n_treated}")
print(f"  处理组城市: {list(treated_baseline['city_name'].unique())}")
print(f"\n对照组样本数: {n_control}")

if n_treated == 0 or n_control == 0:
    raise ValueError(f"基期 {BASELINE_YEAR} 缺少处理组或对照组！")

# ==================== 步骤3: 计算倾向得分 ====================
print("\n" + "=" * 80)
print("步骤3: 计算倾向得分")
print("=" * 80)

# 提取协变量
X_treated = treated_baseline[covariates].values
X_control = control_baseline[covariates].values

# 检查并处理缺失值
if np.isnan(X_treated).any() or np.isnan(X_control).any():
    print("  检测到缺失值，使用均值填充...")
    imputer = SimpleImputer(strategy='mean')
    X_treated = imputer.fit_transform(X_treated)
    X_control = imputer.transform(X_control)

# 合并数据
X = np.vstack([X_treated, X_control])
y = np.hstack([np.ones(n_treated), np.zeros(n_control)])

# 使用Logistic回归计算倾向得分
print("  使用Logistic回归计算倾向得分...")
logit_model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
logit_model.fit(X, y)
pscores = logit_model.predict_proba(X)[:, 1]

pscores_treated = pscores[:n_treated]
pscores_control = pscores[n_treated:]

print(f"\n倾向得分统计:")
print(f"  处理组: 均值={pscores_treated.mean():.4f}, 标准差={pscores_treated.std():.4f}, " +
      f"最小值={pscores_treated.min():.4f}, 最大值={pscores_treated.max():.4f}")
print(f"  对照组: 均值={pscores_control.mean():.4f}, 标准差={pscores_control.std():.4f}, " +
      f"最小值={pscores_control.min():.4f}, 最大值={pscores_control.max():.4f}")

# ==================== 步骤4: 执行近邻匹配 ====================
print("\n" + "=" * 80)
print("步骤4: 执行近邻匹配")
print("=" * 80)
print(f"  匹配方法: 1:{RATIO} 近邻匹配（有放回）")
print(f"  卡尺限制: {CALIPER}")

# 使用NearestNeighbors算法
pscores_control_2d = pscores_control.reshape(-1, 1)
pscores_treated_2d = pscores_treated.reshape(-1, 1)

nbrs = NearestNeighbors(n_neighbors=RATIO, algorithm='ball_tree')
nbrs.fit(pscores_control_2d)

# 找到每个处理组样本的RATIO个最近邻居
distances, indices = nbrs.kneighbors(pscores_treated_2d)

# 应用卡尺限制
matched_pairs = []
discarded_treated = []

for i in range(len(pscores_treated)):
    valid_matches = 0
    for j in range(RATIO):
        dist = distances[i][j]
        idx = indices[i][j]
        if dist <= CALIPER:
            matched_pairs.append({
                'treated_idx': i,
                'control_idx': idx,
                'treated_city': treated_baseline.iloc[i]['city_name'],
                'control_city': control_baseline.iloc[idx]['city_name'],
                'treated_pscore': pscores_treated[i],
                'control_pscore': pscores_control[idx],
                'distance': dist
            })
            valid_matches += 1

    if valid_matches == 0:
        discarded_treated.append(i)

# 统计匹配结果
n_unique_treated_matched = len(set([p['treated_idx'] for p in matched_pairs]))
n_matched_pairs = len(matched_pairs)
n_discarded = len(discarded_treated)
match_rate = n_unique_treated_matched / n_treated * 100

print(f"\n匹配结果统计:")
print(f"  处理组总数: {n_treated}")
print(f"  成功匹配的处理组: {n_unique_treated_matched}")
print(f"  因超出卡尺被丢弃: {n_discarded}")
print(f"  匹配成功率: {match_rate:.2f}%")
print(f"  匹配产生的对照组样本数: {n_matched_pairs}")

if n_unique_treated_matched == 0:
    raise ValueError("没有成功匹配的样本！")

# ==================== 步骤5: 提取匹配成功的城市名单 ====================
print("\n" + "=" * 80)
print("步骤5: 提取匹配成功的城市名单")
print("=" * 80)

# 提取匹配成功的处理组城市
matched_treated_indices = [p['treated_idx'] for p in matched_pairs]
matched_control_indices = [p['control_idx'] for p in matched_pairs]

matched_treated_cities = set(treated_baseline.iloc[matched_treated_indices]['city_name'].values)
matched_control_cities = set(control_baseline.iloc[matched_control_indices]['city_name'].values)

all_matched_cities = matched_treated_cities | matched_control_cities  # 并集

print(f"\n匹配成功的城市统计:")
print(f"  处理组城市数: {len(matched_treated_cities)}")
print(f"  对照组城市数: {len(matched_control_cities)}")
print(f"  城市总数: {len(all_matched_cities)}")

print(f"\n处理组城市列表:")
for i, city in enumerate(sorted(matched_treated_cities), 1):
    print(f"  {i}. {city}")

print(f"\n对照组城市列表 (前20个):")
for i, city in enumerate(sorted(list(matched_control_cities))[:20], 1):
    print(f"  {i}. {city}")
if len(matched_control_cities) > 20:
    print(f"  ... (共{len(matched_control_cities)}个)")

# ==================== 步骤6: 回捞2007-2019年数据 ====================
print("\n" + "=" * 80)
print("步骤6: 回捞2007-2019年数据")
print("=" * 80)

# 从原始数据中提取匹配城市的数据
matched_df = df[df['city_name'].isin(all_matched_cities)].copy()

print(f"\n回捞数据概况:")
print(f"  总样本数: {len(matched_df)}")
print(f"  年份范围: {matched_df['year'].min()} - {matched_df['year'].max()}")
print(f"  处理组样本数: {len(matched_df[matched_df['treat'] == 1])}")
print(f"  对照组样本数: {len(matched_df[matched_df['treat'] == 0])}")

# 检查平衡性
print(f"\n各年份样本数:")
year_counts = matched_df.groupby(['year', 'treat']).size().unstack(fill_value=0)
print(year_counts)

# ==================== 步骤7: 平衡性检验 ====================
print("\n" + "=" * 80)
print("步骤7: 平衡性检验 (基期)")
print("=" * 80)

# 提取基期匹配后的数据
treated_matched = treated_baseline.iloc[matched_treated_indices].copy()
control_matched = control_baseline.iloc[matched_control_indices].copy()

balance_results = []

for var in covariates:
    # 处理组
    t_mean = treated_matched[var].mean()
    t_std = treated_matched[var].std()

    # 对照组
    c_mean = control_matched[var].mean()
    c_std = control_matched[var].std()

    # 标准化偏差
    pooled_std = np.sqrt((t_std**2 + c_std**2) / 2)
    std_bias = (t_mean - c_mean) / pooled_std * 100 if pooled_std > 0 else 0

    # t检验
    t_stat, p_value = stats.ttest_ind(treated_matched[var], control_matched[var])

    balance_results.append({
        'variable': var,
        'treated_mean': t_mean,
        'control_mean': c_mean,
        'std_bias': std_bias,
        't_stat': t_stat,
        'p_value': p_value
    })

balance_df = pd.DataFrame(balance_results)

print(f"\n平衡性检验结果:")
print(f"{'变量':<30} {'处理组均值':<12} {'对照组均值':<12} {'标准化偏差(%)':<15} {'t值':<10} {'P值':<10}")
print("-" * 89)
for row in balance_results:
    sig_mark = '***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.01 else '*' if row['p_value'] < 0.05 else ''
    print(f"{row['variable']:<30} {row['treated_mean']:<12.4f} {row['control_mean']:<12.4f} " +
          f"{row['std_bias']:<15.2f} {row['t_stat']:<10.4f} {row['p_value']:<10.4f} {sig_mark}")

# 判断平衡性
n_balanced = sum(balance_df['p_value'] > 0.05)
print(f"\n平衡性汇总:")
print(f"  平衡变量数 (P>0.05): {n_balanced}/{len(covariates)}")
print(f"  平均绝对标准化偏差: {balance_df['std_bias'].abs().mean():.2f}%")

# ==================== 步骤8: 保存结果 ====================
print("\n" + "=" * 80)
print("步骤8: 保存结果")
print("=" * 80)

# 保存匹配后的完整数据（2007-2019）
output_file = 'baseline_matched_data_2007_2019.xlsx'
matched_df.to_excel(output_file, index=False)
print(f"\n[OK] 已保存匹配后的完整数据: {output_file}")

# 保存城市名单
cities_list_file = 'matched_cities_list.xlsx'
cities_list = pd.DataFrame({
    'city_name': list(all_matched_cities),
    'is_treated': [1 if city in matched_treated_cities else 0 for city in all_matched_cities]
})
cities_list.to_excel(cities_list_file, index=False)
print(f"[OK] 已保存匹配城市名单: {cities_list_file}")

# 保存匹配对信息（仅基期）
matched_pairs_file = 'baseline_matching_pairs.xlsx'
matched_pairs_df = pd.DataFrame(matched_pairs)
matched_pairs_df.to_excel(matched_pairs_file, index=False)
print(f"[OK] 已保存匹配对信息: {matched_pairs_file}")

# 保存平衡性检验结果
balance_file = 'baseline_balance_test.xlsx'
balance_df.to_excel(balance_file, index=False)
print(f"[OK] 已保存平衡性检验结果: {balance_file}")

# ==================== 步骤9: 可视化 ====================
print("\n" + "=" * 80)
print("步骤9: 可视化匹配结果")
print("=" * 80)

# 图1: 倾向得分分布
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 匹配前
axes[0].hist(pscores_treated, bins=20, alpha=0.5, label='处理组', color='red', density=True)
axes[0].hist(pscores_control, bins=20, alpha=0.5, label='对照组', color='blue', density=True)
axes[0].set_xlabel('倾向得分')
axes[0].set_ylabel('密度')
axes[0].set_title(f'匹配前的倾向得分分布 (基期{BASELINE_YEAR})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 匹配后
matched_pscores_control = [p['control_pscore'] for p in matched_pairs]
matched_pscores_treated = [p['treated_pscore'] for p in matched_pairs]

axes[1].hist(matched_pscores_treated, bins=20, alpha=0.5, label='处理组', color='red', density=True)
axes[1].hist(matched_pscores_control, bins=20, alpha=0.5, label='对照组', color='blue', density=True)
axes[1].set_xlabel('倾向得分')
axes[1].set_ylabel('密度')
axes[1].set_title(f'匹配后的倾向得分分布 (基期{BASELINE_YEAR})')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('baseline_pscore_distribution.png', dpi=300, bbox_inches='tight')
print("[OK] 已保存倾向得分分布图: baseline_pscore_distribution.png")
plt.close()

# 图2: 标准化偏差
fig, ax = plt.subplots(figsize=(10, 6))
x_pos = np.arange(len(covariates))
bars = ax.bar(x_pos, balance_df['std_bias'].abs(),
               color=['green' if p > 0.05 else 'red' for p in balance_df['p_value']])
ax.axhline(y=10, color='orange', linestyle='--', label='阈值 (10%)', linewidth=2)
ax.set_xlabel('协变量')
ax.set_ylabel('绝对标准化偏差 (%)')
ax.set_title(f'匹配后的标准化偏差 (基期{BASELINE_YEAR})')
ax.set_xticks(x_pos)
ax.set_xticklabels(covariates, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 在柱状图上标注数值
for i, (bar, bias) in enumerate(zip(bars, balance_df['std_bias'].abs())):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{bias:.1f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('baseline_standardized_bias.png', dpi=300, bbox_inches='tight')
print("[OK] 已保存标准化偏差图: baseline_standardized_bias.png")
plt.close()

print("\n" + "=" * 80)
print("基期匹配PSM分析完成！")
print("=" * 80)
print(f"\n主要输出文件:")
print(f"  1. {output_file} - 匹配后的完整数据(2007-2019)")
print(f"  2. {cities_list_file} - 匹配城市名单")
print(f"  3. {matched_pairs_file} - 基期匹配对信息")
print(f"  4. {balance_file} - 平衡性检验结果")
print(f"  5. baseline_pscore_distribution.png - 倾向得分分布图")
print(f"  6. baseline_standardized_bias.png - 标准化偏差图")
print("\n下一步: 可以使用 baseline_matched_data_2007_2019.xlsx 进行DID分析")
