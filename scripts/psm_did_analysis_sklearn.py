# -*- coding: utf-8 -*-
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
print("PSM-DID 分析 (使用 scikit-learn)")
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

def get_dummies(data, column):
    """创建虚拟变量并删除第一列以避免完全多重共线性"""
    dummies = pd.get_dummies(data[column], prefix=column, drop_first=True)
    return dummies

# 方法1: 简单的DID回归（不含固定效应）
print("\n" + "=" * 80)
print("方法1: 简单DID回归（未包含固定效应）")
print("=" * 80)

# 准备数据
X1 = df[['DID'] + control_vars].astype(float).values
y1 = df['ln_carbon_intensity_ceads'].values

# 拟合模型
model1 = LinearRegression(fit_intercept=True)
model1.fit(X1, y1)

# 预测值和残差
y1_pred = model1.predict(X1)
residuals1 = y1 - y1_pred

# 计算标准误（使用聚类稳健标准误）
# 计算聚类协方差矩阵
def compute_clustered_se(residuals, X, cluster_col):
    """计算聚类稳健标准误"""
    n_params = X.shape[1]
    clusters = df[cluster_col].unique()
    n_clusters = len(clusters)

    # Meat of the sandwich estimator
    meat = np.zeros((n_params, n_params))
    for cluster in clusters:
        cluster_mask = df[cluster_col] == cluster
        X_cluster = X[cluster_mask]
        residuals_cluster = residuals[cluster_mask]
        # 计算每个聚类的得分贡献
        u_cluster = np.dot(X_cluster.T, residuals_cluster)
        u_outer = np.outer(u_cluster, u_cluster)
        meat += u_outer

    # Bread of the sandwich estimator
    XtX = np.dot(X.T, X)
    bread = np.linalg.inv(XtX)

    # Sandwich covariance
    cov = np.dot(bread, np.dot(meat, bread)) * (n_clusters / (n_clusters - 1)) * ((len(X) - 1) / (len(X) - n_params))

    return np.sqrt(np.diag(cov))

# 添加截距项用于计算标准误
X1_with_intercept = np.column_stack([np.ones(len(X1)), X1])
se1 = compute_clustered_se(residuals1, X1_with_intercept, 'city_name')

# DID系数和统计量
did_coef = model1.coef_[0]
did_se = se1[1]  # DID是第二个参数（第一个是截距）
did_t = did_coef / did_se
did_pvalue = 2 * (1 - stats.t.cdf(abs(did_t), df=len(df) - len(control_vars) - 2))

print(f"\n简单DID回归结果:")
print(f"{'变量':<30} {'系数':<12} {'标准误':<12} {'t值':<10} {'P值':<10}")
print("-" * 76)
print(f"{'截距':<30} {model1.intercept_:<12.6f} {se1[0]:<12.6f}")
print(f"{'DID':<30} {did_coef:<12.6f} {did_se:<12.6f} {did_t:<10.4f} {did_pvalue:<10.4f}")
for i, var in enumerate(control_vars):
    print(f"{var:<30} {model1.coef_[i+1]:<12.6f} {se1[i+2]:<12.6f}")

sig_mark1 = '***' if did_pvalue < 0.001 else '**' if did_pvalue < 0.01 else '*' if did_pvalue < 0.05 else '不显著'
print(f"\nDID系数: {did_coef:.6f} (标准误: {did_se:.6f}, t={did_t:.4f}, p={did_pvalue:.4f}) {sig_mark1}")

# 方法2: 包含年份固定效应的DID回归
print("\n" + "=" * 80)
print("方法2: 包含年份固定效应的DID回归")
print("=" * 80)

# 创建年份虚拟变量
year_dummies = get_dummies(df, 'year')

# 准备数据
X2 = pd.concat([df[['DID'] + control_vars], year_dummies], axis=1)
X2_values = X2.astype(float).values
y2 = df['ln_carbon_intensity_ceads'].values

# 拟合模型
model2 = LinearRegression(fit_intercept=True)
model2.fit(X2_values, y2)

# 预测值和残差
y2_pred = model2.predict(X2_values)
residuals2 = y2 - y2_pred

# 计算标准误
X2_with_intercept = np.column_stack([np.ones(len(X2_values)), X2_values])
se2 = compute_clustered_se(residuals2, X2_with_intercept, 'city_name')

# DID系数和统计量
did_coef_fe = model2.coef_[0]
did_se_fe = se2[1]
did_t_fe = did_coef_fe / did_se_fe
did_pvalue_fe = 2 * (1 - stats.t.cdf(abs(did_t_fe), df=len(df) - X2.shape[1] - 2))

print(f"\n年份固定效应回归结果:")
print(f"{'变量':<30} {'系数':<12} {'标准误':<12} {'t值':<10} {'P值':<10}")
print("-" * 76)
print(f"{'截距':<30} {model2.intercept_:<12.6f} {se2[0]:<12.6f}")
print(f"{'DID':<30} {did_coef_fe:<12.6f} {did_se_fe:<12.6f} {did_t_fe:<10.4f} {did_pvalue_fe:<10.4f}")
for i, var in enumerate(control_vars):
    print(f"{var:<30} {model2.coef_[i+1]:<12.6f} {se2[i+2]:<12.6f}")
print(f"... (年份固定效应省略)")

sig_mark2 = '***' if did_pvalue_fe < 0.001 else '**' if did_pvalue_fe < 0.01 else '*' if did_pvalue_fe < 0.05 else '不显著'
print(f"\nDID系数: {did_coef_fe:.6f} (标准误: {did_se_fe:.6f}, t={did_t_fe:.4f}, p={did_pvalue_fe:.4f}) {sig_mark2}")

# 方法3: 包含城市和年份双固定效应的DID回归
print("\n" + "=" * 80)
print("方法3: 包含城市和年份双固定效应的DID回归")
print("=" * 80)

# 计算均值并映射回每个观测值
city_means = df.groupby('city_name')[control_vars + ['ln_carbon_intensity_ceads']].mean().reset_index()
year_means = df.groupby('year')[control_vars + ['ln_carbon_intensity_ceads']].mean().reset_index()
overall_mean = df[control_vars + ['ln_carbon_intensity_ceads']].mean()

# 重命名列以便合并
city_means.columns = ['city_name'] + [var + '_city_mean' for var in control_vars] + ['ln_carbon_intensity_ceads_city_mean']
year_means.columns = ['year'] + [var + '_year_mean' for var in control_vars] + ['ln_carbon_intensity_ceads_year_mean']

# 合并均值
df_demeaned = df.merge(city_means, on='city_name', how='left').merge(year_means, on='year', how='left')

# 去中心化
for var in control_vars + ['ln_carbon_intensity_ceads']:
    df_demeaned[var + '_dm'] = df_demeaned[var] - df_demeaned[var + '_city_mean'] - df_demeaned[var + '_year_mean'] + overall_mean[var]

# 去中心化的DID变量
did_mean = df['DID'].mean()
df_demeaned['DID_dm'] = df['DID'] - did_mean

# 使用去中心化的数据进行回归
X3 = df_demeaned[['DID_dm'] + [var + '_dm' for var in control_vars]].astype(float).values
y3 = df_demeaned['ln_carbon_intensity_ceads_dm'].values

# 拟合模型
model3 = LinearRegression(fit_intercept=True)
model3.fit(X3, y3)

# 预测值和残差
y3_pred = model3.predict(X3)
residuals3 = y3 - y3_pred

# 计算标准误
X3_with_intercept = np.column_stack([np.ones(len(X3)), X3])
se3 = compute_clustered_se(residuals3, X3_with_intercept, 'city_name')

# DID系数和统计量
did_coef_twfe = model3.coef_[0]
did_se_twfe = se3[1]
did_t_twfe = did_coef_twfe / did_se_twfe
did_pvalue_twfe = 2 * (1 - stats.t.cdf(abs(did_t_twfe), df=len(df) - len(control_vars) - 2))

print(f"\n双固定效应回归结果:")
print(f"{'变量':<30} {'系数':<12} {'标准误':<12} {'t值':<10} {'P值':<10}")
print("-" * 76)
print(f"{'截距':<30} {model3.intercept_:<12.6f} {se3[0]:<12.6f}")
print(f"{'DID':<30} {did_coef_twfe:<12.6f} {did_se_twfe:<12.6f} {did_t_twfe:<10.4f} {did_pvalue_twfe:<10.4f}")
for i, var in enumerate(control_vars):
    print(f"{var:<30} {model3.coef_[i+1]:<12.6f} {se3[i+2]:<12.6f}")

sig_mark3 = '***' if did_pvalue_twfe < 0.001 else '**' if did_pvalue_twfe < 0.01 else '*' if did_pvalue_twfe < 0.05 else '不显著'
print(f"\nDID系数: {did_coef_twfe:.6f} (标准误: {did_se_twfe:.6f}, t={did_t_twfe:.4f}, p={did_pvalue_twfe:.4f}) {sig_mark3}")

# 创建汇总表
print("\n" + "=" * 80)
print("DID回归结果汇总")
print("=" * 80)

results_summary = pd.DataFrame({
    '模型': ['简单DID', '年份固定效应', '双固定效应'],
    'DID系数': [did_coef, did_coef_fe, did_coef_twfe],
    '标准误': [did_se, did_se_fe, did_se_twfe],
    't值': [did_t, did_t_fe, did_t_twfe],
    'P值': [did_pvalue, did_pvalue_fe, did_pvalue_twfe],
})

print(results_summary.to_string(index=False))

# 可视化DID系数对比
fig, ax = plt.subplots(figsize=(10, 6))

models = ['简单DID', '年份固定效应', '双固定效应']
coefs = [did_coef, did_coef_fe, did_coef_twfe]
std_errs = [did_se, did_se_fe, did_se_twfe]
pvals = [did_pvalue, did_pvalue_fe, did_pvalue_twfe]

x_pos = np.arange(len(models))
bars = ax.bar(x_pos, coefs, yerr=std_errs, capsize=5, alpha=0.7, color='steelblue', error_kw={'linewidth': 2})

# 添加显著性标记
for i, (coef, pval) in enumerate(zip(coefs, pvals)):
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
    # 创建详细结果表
    results_summary.to_excel(writer, sheet_name='汇总', index=False)

    # 模型1详细结果
    model1_results = pd.DataFrame({
        '变量': ['截距', 'DID'] + control_vars,
        '系数': [model1.intercept_] + list(model1.coef_[:len(control_vars)+1]),
        '标准误': list(se1)
    })
    model1_results.to_excel(writer, sheet_name='简单DID', index=False)

    # 模型2详细结果
    n_model2_coefs = len(control_vars) + 2
    model2_results = pd.DataFrame({
        '变量': ['截距', 'DID'] + control_vars,
        '系数': [model2.intercept_] + list(model2.coef_[:n_model2_coefs-1]),
        '标准误': list(se2[:n_model2_coefs])
    })
    model2_results.to_excel(writer, sheet_name='年份固定效应', index=False)

    # 模型3详细结果
    model3_results = pd.DataFrame({
        '变量': ['截距', 'DID'] + control_vars,
        '系数': [model3.intercept_] + list(model3.coef_[:len(control_vars)+1]),
        '标准误': list(se3)
    })
    model3_results.to_excel(writer, sheet_name='双固定效应', index=False)

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
