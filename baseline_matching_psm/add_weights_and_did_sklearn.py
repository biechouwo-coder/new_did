# -*- coding: utf-8 -*-
"""
为基期匹配数据添加权重并进行PSM-DID分析
=================================================
1. 根据基期匹配信息计算权重
2. 为所有年份的数据添加weight列
3. 使用频次权重进行PSM-DID回归分析
注意：由于频次权重需要整数，我们将使用样本重复的方式实现
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.linear_model import LinearRegression
from scipy import stats

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("步骤1: 为基期匹配数据添加权重")
print("=" * 80)

# 读取匹配后的数据
df = pd.read_excel('baseline_matched_data_2007_2019.xlsx')

# 读取匹配对信息
pairs = pd.read_excel('baseline_matching_pairs.xlsx')

print(f"\n数据概况:")
print(f"  总样本数: {len(df)}")
print(f"  年份范围: {df['year'].min()} - {df['year'].max()}")
print(f"  处理组样本数: {len(df[df['treat'] == 1])}")
print(f"  对照组样本数: {len(df[df['treat'] == 0])}")

print(f"\n匹配对信息:")
print(f"  基期匹配对数量: {len(pairs)}")

# ==================== 计算权重 ====================
print("\n" + "=" * 80)
print("步骤2: 根据基期匹配计算权重")
print("=" * 80)

# 统计每个对照组城市在基期被使用的次数
control_usage = pairs['control_city'].value_counts().to_dict()

print(f"\n对照组城市使用次数统计:")
print(f"  被使用1次: {sum(1 for v in control_usage.values() if v == 1)} 个城市")
print(f"  被使用2次: {sum(1 for v in control_usage.values() if v == 2)} 个城市")
print(f"  被使用3次: {sum(1 for v in control_usage.values() if v == 3)} 个城市")
print(f"  被使用4次: {sum(1 for v in control_usage.values() if v == 4)} 个城市")
print(f"  被使用5次及以上: {sum(1 for v in control_usage.values() if v >= 5)} 个城市")

print(f"\n使用次数最多的5个对照组城市:")
for city, count in pairs['control_city'].value_counts().head(5).items():
    print(f"  {city}: {count}次")

# 为所有数据添加权重
def assign_weight(row):
    """根据城市名和处理组状态分配权重"""
    if row['treat'] == 1:
        # 处理组权重总是1.0
        return 1
    else:
        # 对照组权重 = 使用次数（用于重复样本）
        city = row['city_name']
        usage = control_usage.get(city, 1)
        return usage

df['freq_weight'] = df.apply(assign_weight, axis=1)

print(f"\n权重统计:")
print(f"  处理组权重: {df[df['treat'] == 1]['freq_weight'].unique()}")
print(f"  对照组权重范围: {df[df['treat'] == 0]['freq_weight'].min()} - {df[df['treat'] == 0]['freq_weight'].max()}")

# ==================== 创建加权数据集（通过样本重复） ====================
print("\n" + "=" * 80)
print("步骤3: 创建加权数据集（通过样本重复模拟频次权重）")
print("=" * 80)

# 分离处理组和对照组
treated_df = df[df['treat'] == 1].copy()
control_df = df[df['treat'] == 0].copy()

# 处理组：每个样本重复1次（权重为1）
treated_weighted = treated_df.copy()

# 对照组：根据权重重复样本
control_weighted_list = []
for _, row in control_df.iterrows():
    repeat_times = int(row['freq_weight'])
    # 重复该样本repeat_times次
    for _ in range(repeat_times):
        control_weighted_list.append(row.to_dict())

control_weighted = pd.DataFrame(control_weighted_list)

# 合并处理组和对照组
df_weighted = pd.concat([treated_weighted, control_weighted], ignore_index=True)

print(f"\n加权数据集概况:")
print(f"  原始样本数: {len(df)}")
print(f"  加权后样本数: {len(df_weighted)}")
print(f"  处理组样本数: {len(df_weighted[df_weighted['treat'] == 1])}")
print(f"  对照组样本数: {len(df_weighted[df_weighted['treat'] == 0])}")
print(f"  样本扩增倍数: {len(df_weighted) / len(df):.2f}x")

# 保存加权数据集
output_file = 'baseline_matched_data_weighted_expanded.xlsx'
df_weighted.to_excel(output_file, index=False)
print(f"\n[OK] 已保存加权数据集: {output_file}")

# ==================== PSM-DID分析 ====================
print("\n" + "=" * 80)
print("步骤4: PSM-DID回归分析（使用加权数据）")
print("=" * 80)

# 定义控制变量
control_vars = ['ln_pgdp', 'ln_pop_density', 'industrial_advanced',
                'ln_fdi_openness', 'financial_development']

print(f"\n控制变量:")
for i, var in enumerate(control_vars, 1):
    print(f"  {i}. {var}")

# 计算聚类稳健标准误的函数
def compute_clustered_se(residuals, X, cluster_col):
    """计算聚类稳健标准误"""
    n_params = X.shape[1]
    clusters = df_weighted[cluster_col].unique()
    n_clusters = len(clusters)

    # Meat of the sandwich estimator
    meat = np.zeros((n_params, n_params))
    for cluster in clusters:
        cluster_mask = df_weighted[cluster_col] == cluster
        X_cluster = X[cluster_mask]
        residuals_cluster = residuals[cluster_mask]
        u_cluster = np.dot(X_cluster.T, residuals_cluster)
        u_outer = np.outer(u_cluster, u_cluster)
        meat += u_outer

    # Bread of the sandwich estimator
    XtX = np.dot(X.T, X)
    bread = np.linalg.inv(XtX)

    # Sandwich covariance
    cov = np.dot(bread, np.dot(meat, bread)) * (n_clusters / (n_clusters - 1)) * ((len(X) - 1) / (len(X) - n_params))

    return np.sqrt(np.diag(cov))

# 方法1: 简单DID回归
print("\n" + "=" * 80)
print("方法1: 简单DID回归（使用加权数据）")
print("=" * 80)

X1 = df_weighted[['DID'] + control_vars].astype(float).values
y1 = df_weighted['ln_carbon_intensity_ceads'].values

model1 = LinearRegression(fit_intercept=True)
model1.fit(X1, y1)

y1_pred = model1.predict(X1)
residuals1 = y1 - y1_pred

X1_with_intercept = np.column_stack([np.ones(len(X1)), X1])
se1 = compute_clustered_se(residuals1, X1_with_intercept, 'city_name')

did_coef = model1.coef_[0]
did_se = se1[1]
did_t = did_coef / did_se
did_pvalue = 2 * (1 - stats.t.cdf(abs(did_t), df=len(df_weighted) - len(control_vars) - 2))

sig_mark = '***' if did_pvalue < 0.001 else '**' if did_pvalue < 0.01 else '*' if did_pvalue < 0.05 else ''

print(f"\n简单DID回归结果:")
print(f"{'变量':<30} {'系数':<12} {'标准误':<12} {'t值':<10} {'P值':<10}")
print("-" * 76)
print(f"{'截距':<30} {model1.intercept_:<12.6f} {se1[0]:<12.6f}")
print(f"{'DID':<30} {did_coef:<12.6f} {did_se:<12.6f} {did_t:<10.4f} {did_pvalue:<10.4f} {sig_mark}")
for i, var in enumerate(control_vars):
    print(f"{var:<30} {model1.coef_[i+1]:<12.6f} {se1[i+2]:<12.6f}")

# 方法2: 年份固定效应
print("\n" + "=" * 80)
print("方法2: 年份固定效应（使用加权数据）")
print("=" * 80)

# 创建年份虚拟变量
year_dummies = pd.get_dummies(df_weighted['year'], prefix='year', drop_first=True)
X2 = pd.concat([df_weighted[['DID'] + control_vars], year_dummies], axis=1)
X2_values = X2.astype(float).values
y2 = df_weighted['ln_carbon_intensity_ceads'].values

model2 = LinearRegression(fit_intercept=True)
model2.fit(X2_values, y2)

y2_pred = model2.predict(X2_values)
residuals2 = y2 - y2_pred

X2_with_intercept = np.column_stack([np.ones(len(X2_values)), X2_values])
se2 = compute_clustered_se(residuals2, X2_with_intercept, 'city_name')

did_coef_fe = model2.coef_[0]
did_se_fe = se2[1]
did_t_fe = did_coef_fe / did_se_fe
did_pvalue_fe = 2 * (1 - stats.t.cdf(abs(did_t_fe), df=len(df_weighted) - X2.shape[1] - 2))

sig_mark_fe = '***' if did_pvalue_fe < 0.001 else '**' if did_pvalue_fe < 0.01 else '*' if did_pvalue_fe < 0.05 else ''

print(f"\n年份固定效应回归结果:")
print(f"{'变量':<30} {'系数':<12} {'标准误':<12} {'t值':<10} {'P值':<10}")
print("-" * 76)
print(f"{'截距':<30} {model2.intercept_:<12.6f} {se2[0]:<12.6f}")
print(f"{'DID':<30} {did_coef_fe:<12.6f} {did_se_fe:<12.6f} {did_t_fe:<10.4f} {did_pvalue_fe:<10.4f} {sig_mark_fe}")
for i, var in enumerate(control_vars):
    print(f"{var:<30} {model2.coef_[i+1]:<12.6f} {se2[i+2]:<12.6f}")
print(f"... (年份固定效应省略)")

# 方法3: 双向固定效应（城市 + 年份）
print("\n" + "=" * 80)
print("方法3: 双向固定效应（城市 + 年份）（使用加权数据）")
print("=" * 80)

city_dummies = pd.get_dummies(df_weighted['city_name'], prefix='city', drop_first=True)
X3 = pd.concat([df_weighted[['DID'] + control_vars], city_dummies, year_dummies], axis=1)
X3_values = X3.astype(float).values
y3 = df_weighted['ln_carbon_intensity_ceads'].values

model3 = LinearRegression(fit_intercept=True)
model3.fit(X3_values, y3)

y3_pred = model3.predict(X3_values)
residuals3 = y3 - y3_pred

X3_with_intercept = np.column_stack([np.ones(len(X3_values)), X3_values])
se3 = compute_clustered_se(residuals3, X3_with_intercept, 'city_name')

did_coef_twoway = model3.coef_[0]
did_se_twoway = se3[1]
did_t_twoway = did_coef_twoway / did_se_twoway
did_pvalue_twoway = 2 * (1 - stats.t.cdf(abs(did_t_twoway), df=len(df_weighted) - X3.shape[1] - 2))

sig_mark_twoway = '***' if did_pvalue_twoway < 0.001 else '**' if did_pvalue_twoway < 0.01 else '*' if did_pvalue_twoway < 0.05 else ''

print(f"\n双向固定效应回归结果:")
print(f"{'变量':<30} {'系数':<12} {'标准误':<12} {'t值':<10} {'P值':<10}")
print("-" * 76)
print(f"{'截距':<30} {model3.intercept_:<12.6f} {se3[0]:<12.6f}")
print(f"{'DID':<30} {did_coef_twoway:<12.6f} {did_se_twoway:<12.6f} {did_t_twoway:<10.4f} {did_pvalue_twoway:<10.4f} {sig_mark_twoway}")
for i, var in enumerate(control_vars):
    print(f"{var:<30} {model3.coef_[i+1]:<12.6f} {se3[i+2]:<12.6f}")
print(f"... (城市和年份固定效应省略)")

# ==================== 结果汇总 ====================
print("\n" + "=" * 80)
print("PSM-DID回归结果汇总（使用频次权重）")
print("=" * 80)

results_summary = pd.DataFrame({
    '模型': ['简单DID', '年份固定效应', '双向固定效应'],
    'DID系数': [did_coef, did_coef_fe, did_coef_twoway],
    '标准误': [did_se, did_se_fe, did_se_twoway],
    't值': [did_t, did_t_fe, did_t_twoway],
    'P值': [did_pvalue, did_pvalue_fe, did_pvalue_twoway],
    '显著性': [
        '***' if did_pvalue < 0.001 else '**' if did_pvalue < 0.01 else '*' if did_pvalue < 0.05 else '',
        '***' if did_pvalue_fe < 0.001 else '**' if did_pvalue_fe < 0.01 else '*' if did_pvalue_fe < 0.05 else '',
        '***' if did_pvalue_twoway < 0.001 else '**' if did_pvalue_twoway < 0.01 else '*' if did_pvalue_twoway < 0.05 else ''
    ]
})

print(f"\n{'模型':<20} {'DID系数':<12} {'标准误':<12} {'t值':<10} {'P值':<10} {'显著性':<10}")
print("-" * 76)
for _, row in results_summary.iterrows():
    print(f"{row['模型']:<20} {row['DID系数']:<12.6f} {row['标准误']:<12.6f} " +
          f"{row['t值']:<10.4f} {row['P值']:<10.4f} {row['显著性']:<10}")

# 保存结果汇总
results_file = 'psm_did_results_weighted.xlsx'
results_summary.to_excel(results_file, index=False)
print(f"\n[OK] 已保存结果汇总: {results_file}")

# ==================== 可视化 ====================
print("\n" + "=" * 80)
print("步骤5: 可视化回归结果")
print("=" * 80)

# 图1: DID系数对比
fig, ax = plt.subplots(figsize=(10, 6))

x_pos = np.arange(len(results_summary))
coefs = results_summary['DID系数'].values
ses = results_summary['标准误'].values

bars = ax.bar(x_pos, coefs, yerr=ses, capsize=5,
              color=['skyblue', 'lightcoral', 'lightgreen'],
              alpha=0.7, edgecolor='black', linewidth=1.5)

# 添加显著性标记
for i, (bar, pval) in enumerate(zip(bars, results_summary['P值'])):
    height = bar.get_height()
    sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
    ax.text(bar.get_x() + bar.get_width()/2, height + ses[i] + 0.01,
            sig, ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

ax.set_xlabel('模型类型', fontsize=12)
ax.set_ylabel('DID系数', fontsize=12)
ax.set_title('PSM-DID回归系数对比（基期匹配 + 频次权重）', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(results_summary['模型'], rotation=15, ha='right')
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标注
for i, (bar, coef, se) in enumerate(zip(bars, coefs, ses)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - se - 0.02,
            f'{coef:.4f}', ha='center', va='top', fontsize=10)

plt.tight_layout()
plt.savefig('baseline_psm_did_coefficients_weighted.png', dpi=300, bbox_inches='tight')
print("[OK] 已保存DID系数对比图: baseline_psm_did_coefficients_weighted.png")
plt.close()

# 图2: DID系数及95%置信区间
fig, ax = plt.subplots(figsize=(12, 6))

conf_intervals = [(coef - 1.96*se, coef + 1.96*se)
                  for coef, se in zip(coefs, ses)]

x_pos = np.arange(len(results_summary))
colors = ['skyblue', 'lightcoral', 'lightgreen']

for i, ((ci_lower, ci_upper), color) in enumerate(zip(conf_intervals, colors)):
    ax.bar(x_pos[i], coefs[i], width=0.6,
           color=color, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.errorbar(x_pos[i], coefs[i], yerr=[[coefs[i]-ci_lower], [ci_upper-coefs[i]]],
                fmt='none', ecolor='black', elinewidth=2, capsize=8)

# 添加显著性标记
for i, pval in enumerate(results_summary['P值']):
    sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
    y_pos = conf_intervals[i][1] + 0.02
    ax.text(x_pos[i], y_pos, sig, ha='center', va='bottom',
            fontsize=14, fontweight='bold')

# 添加数值标注
for i, (coef, (ci_lower, ci_upper)) in enumerate(zip(coefs, conf_intervals)):
    ax.text(x_pos[i], coef, f'{coef:.4f}\n[{ci_lower:.4f}, {ci_upper:.4f}]',
            ha='center', va='center', fontsize=10, fontweight='bold')

ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
ax.set_xlabel('模型类型', fontsize=12)
ax.set_ylabel('DID系数', fontsize=12)
ax.set_title('PSM-DID回归系数及95%置信区间（基期匹配 + 频次权重）',
             fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(results_summary['模型'], rotation=15, ha='right')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('baseline_psm_did_coefficients_with_ci_weighted.png', dpi=300, bbox_inches='tight')
print("[OK] 已保存DID系数及置信区间图: baseline_psm_did_coefficients_with_ci_weighted.png")
plt.close()

print("\n" + "=" * 80)
print("PSM-DID分析完成！")
print("=" * 80)
print(f"\n主要输出文件:")
print(f"  1. {output_file} - 加权数据集（样本重复）")
print(f"  2. {results_file} - 回归结果汇总")
print(f"  3. baseline_psm_did_coefficients_weighted.png - DID系数对比图")
print(f"  4. baseline_psm_did_coefficients_with_ci_weighted.png - DID系数及置信区间图")
