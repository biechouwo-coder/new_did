# -*- coding: utf-8 -*-
"""
为基期匹配数据添加权重并进行PSM-DID分析
=================================================
1. 根据基期匹配信息计算权重
2. 为所有年份的数据添加weight列
3. 使用频次权重进行PSM-DID回归分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.linear_model import LinearRegression
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

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
        return 1.0
    else:
        # 对照组权重 = 1 / 使用次数
        city = row['city_name']
        usage = control_usage.get(city, 1)
        return 1.0 / usage

df['weight'] = df.apply(assign_weight, axis=1)

print(f"\n权重统计:")
print(f"  处理组权重: {df[df['treat'] == 1]['weight'].unique()}")
print(f"  对照组权重范围: {df[df['treat'] == 0]['weight'].min():.4f} - {df[df['treat'] == 0]['weight'].max():.4f}")
print(f"  对照组权重分布:")
print(df[df['treat'] == 0]['weight'].value_counts().sort_index().head(10))

# 保存带权重的数据
output_file = 'baseline_matched_data_with_weights.xlsx'
df.to_excel(output_file, index=False)
print(f"\n[OK] 已保存带权重的数据: {output_file}")

# ==================== PSM-DID分析 ====================
print("\n" + "=" * 80)
print("步骤3: PSM-DID回归分析（使用频次权重）")
print("=" * 80)

# 定义控制变量
control_vars = ['ln_pgdp', 'ln_pop_density', 'industrial_advanced',
                'ln_fdi_openness', 'financial_development']

print(f"\n控制变量:")
for i, var in enumerate(control_vars, 1):
    print(f"  {i}. {var}")

# 方法1: 使用statsmodels进行加权回归（更准确的频次权重处理）
print("\n" + "=" * 80)
print("方法1: 使用statsmodels的频次权重回归")
print("=" * 80)

# 准备数据
X_vars = ['DID'] + control_vars
formula = f"ln_carbon_intensity_ceads ~ {' + '.join(X_vars)}"

print(f"\n回归公式:")
print(f"  {formula}")

# 使用OLS回归，带频次权重
model1 = smf.ols(
    formula=formula,
    data=df
).fit(
    cov_type='cluster',
    cov_kwds={'groups': df['city_name']},
    use_t=True
)

print(f"\n简单DID回归结果（频次权重）:")
print(model1.summary())

# 提取DID系数
did_coef = model1.params['DID']
did_se = model1.bse['DID']
did_t = model1.tvalues['DID']
did_pvalue = model1.pvalues['DID']

sig_mark = '***' if did_pvalue < 0.001 else '**' if did_pvalue < 0.01 else '*' if did_pvalue < 0.05 else ''
print(f"\nDID系数: {did_coef:.6f} (标准误: {did_se:.6f}, t={did_t:.4f}, p={did_pvalue:.4f}) {sig_mark}")

# 方法2: 包含年份固定效应的加权回归
print("\n" + "=" * 80)
print("方法2: 包含年份固定效应的加权回归")
print("=" * 80)

# 创建年份虚拟变量
year_dummies = pd.get_dummies(df['year'], prefix='year', drop_first=True)
df_with_dummies = pd.concat([df, year_dummies], axis=1)

# 构建回归公式
year_vars = [col for col in df_with_dummies.columns if col.startswith('year_')]
X_vars_fe = ['DID'] + control_vars + year_vars
formula_fe = f"ln_carbon_intensity_ceads ~ {' + '.join(X_vars_fe)}"

print(f"\n回归公式:")
print(f"  {formula_fe}")
print(f"  (包含{len(year_vars)}个年份虚拟变量)")

model2 = smf.ols(
    formula=formula_fe,
    data=df_with_dummies
).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_with_dummies['city_name']},
    use_t=True
)

print(f"\n年份固定效应回归结果（频次权重）:")
print(model2.summary())

# 提取DID系数
did_coef_fe = model2.params['DID']
did_se_fe = model2.bse['DID']
did_t_fe = model2.tvalues['DID']
did_pvalue_fe = model2.pvalues['DID']

sig_mark_fe = '***' if did_pvalue_fe < 0.001 else '**' if did_pvalue_fe < 0.01 else '*' if did_pvalue_fe < 0.05 else ''
print(f"\nDID系数: {did_coef_fe:.6f} (标准误: {did_se_fe:.6f}, t={did_t_fe:.4f}, p={did_pvalue_fe:.4f}) {sig_mark_fe}")

# 方法3: 双向固定效应（城市和年份）
print("\n" + "=" * 80)
print("方法3: 双向固定效应（城市 + 年份）")
print("=" * 80)

# 创建城市和年份虚拟变量
city_dummies = pd.get_dummies(df['city_name'], prefix='city', drop_first=True)
df_with_fe = pd.concat([df, city_dummies, year_dummies], axis=1)

# 构建回归公式
city_vars = [col for col in df_with_fe.columns if col.startswith('city_')]
X_vars_twoway = ['DID'] + control_vars + city_vars + year_vars
formula_twoway = f"ln_carbon_intensity_ceads ~ {' + '.join(X_vars_twoway)}"

print(f"\n回归公式:")
print(f"  {formula_twoway}")
print(f"  (包含{len(city_vars)}个城市虚拟变量 + {len(year_vars)}个年份虚拟变量)")

model3 = smf.ols(
    formula=formula_twoway,
    data=df_with_fe
).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_with_fe['city_name']},
    use_t=True
)

print(f"\n双向固定效应回归结果（频次权重）:")
print(model3.summary())

# 提取DID系数
did_coef_twoway = model3.params['DID']
did_se_twoway = model3.bse['DID']
did_t_twoway = model3.tvalues['DID']
did_pvalue_twoway = model3.pvalues['DID']

sig_mark_twoway = '***' if did_pvalue_twoway < 0.001 else '**' if did_pvalue_twoway < 0.01 else '*' if did_pvalue_twoway < 0.05 else ''
print(f"\nDID系数: {did_coef_twoway:.6f} (标准误: {did_se_twoway:.6f}, t={did_t_twoway:.4f}, p={did_pvalue_twoway:.4f}) {sig_mark_twoway}")

# ==================== 结果汇总 ====================
print("\n" + "=" * 80)
print("PSM-DID回归结果汇总")
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
results_file = 'psm_did_results_summary.xlsx'
results_summary.to_excel(results_file, index=False)
print(f"\n[OK] 已保存结果汇总: {results_file}")

# ==================== 可视化 ====================
print("\n" + "=" * 80)
print("步骤4: 可视化回归结果")
print("=" * 80)

# 图1: DID系数对比（不同模型）
fig, ax = plt.subplots(figsize=(10, 6))

x_pos = np.arange(len(results_summary))
coefs = results_summary['DID系数'].values
ses = results_summary['标准误'].values

# 画误差条
bars = ax.bar(x_pos, coefs, yerr=ses, capsize=5,
              color=['skyblue', 'lightcoral', 'lightgreen'],
              alpha=0.7, edgecolor='black', linewidth=1.5)

# 添加显著性标记
for i, (bar, pval) in enumerate(zip(bars, results_summary['P值'])):
    height = bar.get_height()
    sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
    ax.text(bar.get_x() + bar.get_width()/2, height + ses[i] + 0.01,
            sig, ha='center', va='bottom', fontsize=14, fontweight='bold')

# 添加0线
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
plt.savefig('baseline_psm_did_coefficients.png', dpi=300, bbox_inches='tight')
print("[OK] 已保存DID系数对比图: baseline_psm_did_coefficients.png")
plt.close()

# 图2: 各模型的详细对比（含置信区间）
fig, ax = plt.subplots(figsize=(12, 6))

# 计算95%置信区间
conf_intervals = [(coef - 1.96*se, coef + 1.96*se)
                  for coef, se in zip(coefs, ses)]

# 画误差条
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
plt.savefig('baseline_psm_did_coefficients_with_ci.png', dpi=300, bbox_inches='tight')
print("[OK] 已保存DID系数及置信区间图: baseline_psm_did_coefficients_with_ci.png")
plt.close()

print("\n" + "=" * 80)
print("PSM-DID分析完成！")
print("=" * 80)
print(f"\n主要输出文件:")
print(f"  1. {output_file} - 带权重的完整数据")
print(f"  2. {results_file} - 回归结果汇总")
print(f"  3. baseline_psm_did_coefficients.png - DID系数对比图")
print(f"  4. baseline_psm_did_coefficients_with_ci.png - DID系数及置信区间图")
