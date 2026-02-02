# -*- coding: utf-8 -*-
"""
完整的PSM-DID分析流程（使用DID_lag1变量）
所有结果保存到results_lag1/文件夹
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from scipy import stats
from matplotlib import rcParams
from collections import Counter

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("PSM-DID 完整分析（使用 DID_lag1）")
print("=" * 80)

# ========== 步骤1: PSM匹配 ==========
print("\n" + "=" * 80)
print("步骤1: 倾向得分匹配 (PSM)")
print("=" * 80)

# 读取数据（带DID_lag1）
df = pd.read_excel('CEADs_最终数据集_2007-2019_V2_插值版_带DID_lag1_缩尾版.xlsx')

# 用DID_lag1替换DID变量用于匹配
df['DID_original'] = df['DID']
df['DID'] = df['DID_lag1']

# 定义匹配变量
covariates = ['ln_pgdp', 'ln_pop_density', 'industrial_advanced', 'ln_fdi_openness', 'financial_development']

print(f"\n匹配变量: {covariates}")

# 获取年份列表
years = sorted(df['year'].unique())
print(f"年份范围: {years[0]} - {years[-1]} (共{len(years)}年)")

# 初始化存储
all_matched_data = []

# 按年份进行PSM
for year in years:
    print(f"\n{'='*80}")
    print(f"年份: {year}")
    print(f"{'='*80}")

    df_year = df[df['year'] == year].copy()
    treated = df_year[df_year['treat'] == 1].copy()
    control = df_year[df_year['treat'] == 0].copy()

    n_treated = len(treated)
    n_control = len(control)

    print(f"处理组: {n_treated}, 对照组: {n_control}")

    if n_treated == 0 or n_control == 0:
        continue

    # 估计倾向得分
    X = control[covariates].values
    y = control['treat'].values

    if len(X) > 10:
        try:
            log_model = LogisticRegression(solver='lbfgs', max_iter=1000)
            log_model.fit(X, y)
            pscores_control = log_model.predict_proba(X)[:, 1]
            pscores_treated = log_model.predict_proba(treated[covariates].values)[:, 1]
        except:
            pscores_control = np.zeros(len(control))
            pscores_treated = np.ones(len(treated))
    else:
        pscores_control = np.zeros(len(control))
        pscores_treated = np.ones(len(treated))

    # 执行1:2匹配
    CALIPER = 0.05

    pscores_control_2d = pscores_control.reshape(-1, 1)
    pscores_treated_2d = pscores_treated.reshape(-1, 1)

    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree')
    nbrs.fit(pscores_control_2d)

    distances, indices = nbrs.kneighbors(pscores_treated_2d)

    matched_pairs = []

    for i in range(len(pscores_treated)):
        for j in range(2):
            dist = distances[i][j]
            idx = indices[i][j]
            if dist <= CALIPER:
                matched_pairs.append({
                    'treated_idx': i,
                    'control_idx': idx,
                })

    if len(matched_pairs) == 0:
        continue

    # 提取匹配数据
    matched_treated_indices = [p['treated_idx'] for p in matched_pairs]
    matched_control_indices = [p['control_idx'] for p in matched_pairs]

    treated_matched = treated.iloc[matched_treated_indices].copy()
    control_matched = control.iloc[matched_control_indices].copy()

    # 计算权重
    control_usage = Counter(matched_control_indices)
    treated_matched['weight'] = 1.0
    control_matched['weight'] = control_matched.index.map(
        lambda idx: 1.0 / control_usage.get(control.index.get_loc(idx), 1)
    )

    year_matched = pd.concat([treated_matched, control_matched], ignore_index=True)
    all_matched_data.append(year_matched)

# 合并所有年份
df_matched = pd.concat(all_matched_data, ignore_index=True)
print(f"\n匹配后总样本数: {len(df_matched)}")

# 去重
df_dedup = df_matched.drop_duplicates(subset=['city_name', 'year'], keep='first')
print(f"去重后样本数: {len(df_dedup)}")

# 保存匹配数据
df_dedup.to_excel('results_lag1/PSM_matched_data_dedup_lag1.xlsx', index=False)
print("匹配数据已保存: results_lag1/PSM_matched_data_dedup_lag1.xlsx")

# ========== 步骤2: DID回归分析 ==========
print("\n" + "=" * 80)
print("步骤2: DID回归分析")
print("=" * 80)

# 读取去重后的匹配数据
df = df_dedup

control_vars = covariates

def get_dummies(data, column):
    dummies = pd.get_dummies(data[column], prefix=column, drop_first=True)
    return dummies

def compute_clustered_se(residuals, X, cluster_col, df_ref):
    n_params = X.shape[1]
    clusters = df_ref[cluster_col].unique()
    n_clusters = len(clusters)

    meat = np.zeros((n_params, n_params))
    for cluster in clusters:
        cluster_mask = df_ref[cluster_col] == cluster
        X_cluster = X[cluster_mask]
        residuals_cluster = residuals[cluster_mask]
        u_cluster = np.dot(X_cluster.T, residuals_cluster)
        u_outer = np.outer(u_cluster, u_cluster)
        meat += u_outer

    XtX = np.dot(X.T, X)
    bread = np.linalg.inv(XtX)

    cov = np.dot(bread, np.dot(meat, bread)) * (n_clusters / (n_clusters - 1)) * ((len(X) - 1) / (len(X) - n_params))
    return np.sqrt(np.diag(cov))

# 模型1: 简单DID
print("\n模型1: 简单DID回归")
X1 = df[['DID'] + control_vars].astype(float).values
y1 = df['ln_carbon_intensity_ceads'].values

model1 = LinearRegression(fit_intercept=True)
model1.fit(X1, y1)
y1_pred = model1.predict(X1)
residuals1 = y1 - y1_pred

X1_with_intercept = np.column_stack([np.ones(len(X1)), X1])
se1 = compute_clustered_se(residuals1, X1_with_intercept, 'city_name', df)

did_coef1 = model1.coef_[0]
did_se1 = se1[1]
did_t1 = did_coef1 / did_se1
did_p1 = 2 * (1 - stats.t.cdf(abs(did_t1), df=len(df) - len(control_vars) - 2))

print(f"DID系数: {did_coef1:.6f} (SE={did_se1:.6f}, t={did_t1:.4f}, p={did_p1:.4f})")

# 模型2: 年份固定效应
print("\n模型2: 年份固定效应DID回归")
year_dummies = get_dummies(df, 'year')
X2 = pd.concat([df[['DID'] + control_vars], year_dummies], axis=1)
X2_values = X2.astype(float).values
y2 = df['ln_carbon_intensity_ceads'].values

model2 = LinearRegression(fit_intercept=True)
model2.fit(X2_values, y2)
y2_pred = model2.predict(X2_values)
residuals2 = y2 - y2_pred

X2_with_intercept = np.column_stack([np.ones(len(X2_values)), X2_values])
se2 = compute_clustered_se(residuals2, X2_with_intercept, 'city_name', df)

did_coef2 = model2.coef_[0]
did_se2 = se2[1]
did_t2 = did_coef2 / did_se2
did_p2 = 2 * (1 - stats.t.cdf(abs(did_t2), df=len(df) - X2.shape[1] - 2))

print(f"DID系数: {did_coef2:.6f} (SE={did_se2:.6f}, t={did_t2:.4f}, p={did_p2:.4f})")

# 模型3: 双固定效应
print("\n模型3: 双固定效应DID回归")
city_means = df.groupby('city_name')[control_vars + ['ln_carbon_intensity_ceads']].mean().reset_index()
year_means = df.groupby('year')[control_vars + ['ln_carbon_intensity_ceads']].mean().reset_index()
overall_mean = df[control_vars + ['ln_carbon_intensity_ceads']].mean()

city_means.columns = ['city_name'] + [var + '_city_mean' for var in control_vars] + ['ln_carbon_intensity_ceads_city_mean']
year_means.columns = ['year'] + [var + '_year_mean' for var in control_vars] + ['ln_carbon_intensity_ceads_year_mean']

df_demeaned = df.merge(city_means, on='city_name', how='left').merge(year_means, on='year', how='left')

for var in control_vars + ['ln_carbon_intensity_ceads']:
    df_demeaned[var + '_dm'] = df_demeaned[var] - df_demeaned[var + '_city_mean'] - df_demeaned[var + '_year_mean'] + overall_mean[var]

did_mean = df['DID'].mean()
df_demeaned['DID_dm'] = df['DID'] - did_mean

X3 = df_demeaned[['DID_dm'] + [var + '_dm' for var in control_vars]].astype(float).values
y3 = df_demeaned['ln_carbon_intensity_ceads_dm'].values

model3 = LinearRegression(fit_intercept=True)
model3.fit(X3, y3)
y3_pred = model3.predict(X3)
residuals3 = y3 - y3_pred

X3_with_intercept = np.column_stack([np.ones(len(X3)), X3])
se3 = compute_clustered_se(residuals3, X3_with_intercept, 'city_name', df_demeaned)

did_coef3 = model3.coef_[0]
did_se3 = se3[1]
did_t3 = did_coef3 / did_se3
did_p3 = 2 * (1 - stats.t.cdf(abs(did_t3), df=len(df) - len(control_vars) - 2))

print(f"DID系数: {did_coef3:.6f} (SE={did_se3:.6f}, t={did_t3:.4f}, p={did_p3:.4f})")

# ========== 步骤3: 保存结果 ==========
print("\n" + "=" * 80)
print("步骤3: 保存结果")
print("=" * 80)

# 汇总表
results_summary = pd.DataFrame({
    '模型': ['简单DID', '年份固定效应', '双固定效应'],
    'DID系数': [did_coef1, did_coef2, did_coef3],
    '标准误': [did_se1, did_se2, did_se3],
    't值': [did_t1, did_t2, did_t3],
    'P值': [did_p1, did_p2, did_p3],
    '显著性': ['***' if did_p1 < 0.001 else '**' if did_p1 < 0.01 else '*' if did_p1 < 0.05 else '不显著',
               '***' if did_p2 < 0.001 else '**' if did_p2 < 0.01 else '*' if did_p2 < 0.05 else '不显著',
               '***' if did_p3 < 0.001 else '**' if did_p3 < 0.01 else '*' if did_p3 < 0.05 else '不显著']
})

print("\nDID回归结果汇总:")
print(results_summary.to_string(index=False))

# 保存到Excel
with pd.ExcelWriter('results_lag1/PSM_DID_regression_results_lag1.xlsx', engine='openpyxl') as writer:
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

print("\n回归结果已保存: results_lag1/PSM_DID_regression_results_lag1.xlsx")

# 生成对比图
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))

models = ['简单DID', '年份固定效应', '双固定效应']
coefs = [did_coef1, did_coef2, did_coef3]
std_errs = [did_se1, did_se2, did_se3]
pvals = [did_p1, did_p2, did_p3]

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
ax.set_title('DID_lag1: 不同模型的系数对比', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(models, fontsize=10)
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('results_lag1/DID_lag1_coefficients_comparison.png', dpi=300, bbox_inches='tight')
print("系数对比图已保存: results_lag1/DID_lag1_coefficients_comparison.png")

print(f"\n{'=' * 80}")
print("PSM-DID分析完成（使用DID_lag1）！")
print(f"{'=' * 80}")
print("\n所有结果已保存到 results_lag1/ 文件夹")
print("1. PSM_matched_data_dedup_lag1.xlsx - 匹配数据")
print("2. PSM_DID_regression_results_lag1.xlsx - 回归结果")
print("3. DID_lag1_coefficients_comparison.png - 系数对比图")
