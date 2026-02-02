# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
# import seaborn as sns  # Not available, not used

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("PSM-DID 分析")
print("=" * 80)

# 读取去重后的PSM匹配数据
df = pd.read_excel('CEADs_PSM_matched_data_dedup.xlsx')

print(f"\n数据概况:")
print(f"总样本数: {len(df)}")
print(f"城市数量: {df['city_name'].nunique()}")
print(f"年份范围: {df['year'].min()} - {df['year'].max()}")
print(f"\n列名: {list(df.columns)}")

# 定义控制变量
control_vars = ['ln_pgdp', 'ln_pop_density', 'industrial_advanced', 'ln_fdi_openness', 'financial_development']

print(f"\n控制变量:")
for i, var in enumerate(control_vars, 1):
    print(f"{i}. {var}")

# 定义模型公式
# DID模型: Y = α + β*DID + γ*X + δ_t + λ_i + ε
# 其中: X是控制变量，δ_t是年份固定效应，λ_i是城市固定效应

# 方法1: 简单的DID回归（不含固定效应）
print("\n" + "=" * 80)
print("方法1: 简单DID回归（未包含固定效应）")
print("=" * 80)

import statsmodels.formula.api as sm

# 模型1: Y = α + β*DID + γ1*X1 + γ2*X2 + ... + ε
formula_ols = 'ln_carbon_intensity_ceads ~ DID + ' + ' + '.join(control_vars)

print(f"\n模型公式:")
print(formula_ols)

model_ols = sm.ols(formula_ols, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['city_name']})

print("\n回归结果（简单DID）:")
print(model_ols.summary())

# 提取DID系数和统计信息
did_coef = model_ols.params['DID']
did_pvalue = model_ols.pvalues['DID']
did_std_err = model_ols.bse['DID']

print(f"\nDID系数统计:")
print(f"系数: {did_coef:.6f}")
print(f"标准误: {did_std_err:.6f}")
print(f"t统计量: {did_coef/did_std_err:.4f}")
print(f"P值: {did_pvalue:.4f}")
print(f"显著性: {'***' if did_pvalue < 0.001 else '**' if did_pvalue < 0.01 else '*' if did_pvalue < 0.05 else '不显著'}")

# 方法2: 包含年份固定效应的DID回归
print("\n" + "=" * 80)
print("方法2: 包含年份固定效应的DID回归")
print("=" * 80)

formula_fe = 'ln_carbon_intensity_ceads ~ DID + ' + ' + '.join(control_vars) + ' + C(year)'

print(f"\n模型公式:")
print(formula_fe)

model_fe = sm.ols(formula_fe, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['city_name']})

print("\n回归结果（年份固定效应）:")
print(model_fe.summary())

# 提取DID系数
did_coef_fe = model_fe.params['DID']
did_pvalue_fe = model_fe.pvalues['DID']
did_std_err_fe = model_fe.bse['DID']

print(f"\nDID系数统计:")
print(f"系数: {did_coef_fe:.6f}")
print(f"标准误: {did_std_err_fe:.6f}")
print(f"t统计量: {did_coef_fe/did_std_err_fe:.4f}")
print(f"P值: {did_pvalue_fe:.4f}")
print(f"显著性: {'***' if did_pvalue_fe < 0.001 else '**' if did_pvalue_fe < 0.01 else '*' if did_pvalue_fe < 0.05 else '不显著'}")

# 方法3: 包含城市和年份双固定效应的DID回归
print("\n" + "=" * 80)
print("方法3: 包含城市和年份双固定效应的DID回归")
print("=" * 80)

# 创建实体效应模型使用 demeaning 方法
# Y_it - Y_bar_i - Y_bar_t + Y_bar_bar

# 计算均值
city_means = df.groupby('city_name')[control_vars + ['ln_carbon_intensity_ceads']].mean()
year_means = df.groupby('year')[control_vars + ['ln_carbon_intensity_ceads']].mean()
overall_mean = df[control_vars + ['ln_carbon_intensity_ceads']].mean()

# 去中心化
df_demeaned = df.copy()
for var in control_vars + ['ln_carbon_intensity_ceads']:
    df_demeaned[var + '_demeaned'] = df[var] - city_means[var] - year_means[var.replace('ln_', '')] + overall_mean[var]

# 去中心化的DID变量
did_mean = df['DID'].mean()
df_demeaned['DID_demeaned'] = df['DID'] - did_mean

# 使用去中心化的数据进行回归
formula_demeaned = 'ln_carbon_intensity_ceads_demeaned ~ DID_demeaned + ' + ' + '.join([var + '_demeaned' for var in control_vars])

print(f"\n模型公式（去中心化）:")
print(formula_demeaned)

model_demeaned = sm.ols(formula_demeaned, data=df_demeaned).fit()

print("\n回归结果（双固定效应）:")
print(model_demeaned.summary())

# 提取DID系数
did_coef_twfe = model_demeaned.params['DID_demeaned']
did_pvalue_twfe = model_demeaned.pvalues['DID_demeaned']
did_std_err_twfe = model_demeaned.bse['DID_demeaned']

print(f"\nDID系数统计:")
print(f"系数: {did_coef_twfe:.6f}")
print(f"标准误: {did_std_err_twfe:.6f}")
print(f"t统计量: {did_coef_twfe/did_std_err_twfe:.4f}")
print(f"P值: {did_pvalue_twfe:.4f}")
print(f"显著性: {'***' if did_pvalue_twfe < 0.001 else '**' if did_pvalue_twfe < 0.01 else '*' if did_pvalue_twfe < 0.05 else '不显著'}")

# 创建汇总表
print("\n" + "=" * 80)
print("DID回归结果汇总")
print("=" * 80)

results_summary = pd.DataFrame({
    '模型': ['简单DID', '年份固定效应', '双固定效应'],
    'DID系数': [did_coef, did_coef_fe, did_coef_twfe],
    '标准误': [did_std_err, did_std_err_fe, did_std_err_twfe],
    't值': [did_coef/did_std_err, did_coef_fe/did_std_err_fe, did_coef_twfe/did_std_err_twfe],
    'P值': [did_pvalue, did_pvalue_fe, did_pvalue_twfe],
})

print(results_summary.to_string(index=False))

# 可视化DID系数对比
fig, ax = plt.subplots(figsize=(10, 6))

models = ['简单DID', '年份固定效应', '双固定效应']
coefs = [did_coef, did_coef_fe, did_coef_twfe]
std_errs = [did_std_err, did_std_err_fe, did_std_err_twfe]

x_pos = np.arange(len(models))
bars = ax.bar(x_pos, coefs, yerr=std_errs, capsize=5, alpha=0.7, color='steelblue', error_kw={'linewidth': 2})

# 添加显著性标记
for i, (coef, pval) in enumerate(zip(coefs, [did_pvalue, did_pvalue_fe, did_pvalue_twfe])):
    sig_mark = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
    if sig_mark:
        ax.text(i, coef + std_errs[i] + 0.02, sig_mark, fontsize=16, ha='center', va='bottom', color='red')

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax.set_xlabel('模型类型', fontsize=12)
ax.set_ylabel('DID系数', fontsize=12)
ax.set_title('不同DID模型的系数对比', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(models, fontsize=10)
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('PSM_DID_coefficients_comparison.png', dpi=300, bbox_inches='tight')
print("\n系数对比图已保存: PSM_DID_coefficients_comparison.png")

# 保存回归结果到Excel
with pd.ExcelWriter('PSM_DID_regression_results.xlsx', engine='openpyxl') as writer:
    model_ols.summary.to_excel(writer, sheet_name='简单DID', index=True)
    model_fe.summary.to_excel(writer, sheet_name='年份固定效应', index=True)
    model_demeaned.summary.to_excel(writer, sheet_name='双固定效应', index=True)
    results_summary.to_excel(writer, sheet_name='汇总', index=False)

print("\n回归结果已保存到: PSM_DID_regression_results.xlsx")

print(f"\n{'=' * 80}")
print("PSM-DID分析完成！")
print(f"{'=' * 80}")
print(f"\n输出文件:")
print(f"1. PSM_DID_regression_results.xlsx - 完整回归结果")
print(f"2. PSM_DID_coefficients_comparison.png - 系数对比图")

print(f"\n主要发现:")
print(f"在控制了{len(control_vars)}个变量后，低碳试点政策对碳强度的净效应为:")
for idx, row in results_summary.iterrows():
    sig = '***' if row['P值'] < 0.001 else '**' if row['P值'] < 0.01 else '*' if row['P值'] < 0.05 else '不显著'
    print(f"  {row['模型']}: {row['DID系数']:.6f} {sig}")
