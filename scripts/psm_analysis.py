# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("倾向得分匹配（PSM）分析")
print("=" * 80)

# 读取数据
df = pd.read_excel('CEADs_最终数据集_2007-2019_V2_插值版_带DID_treat.xlsx')

# 定义匹配变量（协变量）
covariates = ['ln_pgdp', 'ln_pop_density', 'industrial_advanced', 'ln_fdi_openness', 'financial_development']

print(f"\n匹配变量（协变量）:")
for i, var in enumerate(covariates, 1):
    print(f"{i}. {var}")

# 获取年份列表
years = sorted(df['year'].unique())
print(f"\n年份范围: {years[0]} - {years[-1]} (共{len(years)}年)")

# 存储每年的匹配结果
matched_results = []
balance_test_results = []

# 对每一年进行匹配
for year in years:
    print(f"\n{'=' * 80}")
    print(f"处理年份: {year}")
    print(f"{'=' * 80}")

    # 提取当年数据
    year_data = df[df['year'] == year].copy()

    # 分离处理组和对照组
    treated = year_data[year_data['treat'] == 1].copy()
    control = year_data[year_data['treat'] == 0].copy()

    n_treated = len(treated)
    n_control = len(control)

    print(f"处理组样本数: {n_treated}")
    print(f"对照组样本数: {n_control}")

    if n_treated == 0 or n_control == 0:
        print(f"警告：年份 {year} 缺少处理组或对照组，跳过该年")
        continue

    # 提取协变量
    X_treated = treated[covariates].values
    X_control = control[covariates].values

    # 检查是否有缺失值
    if np.isnan(X_treated).any() or np.isnan(X_control).any():
        print(f"警告：年份 {year} 存在缺失值，使用均值填充")
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        X_treated = imputer.fit_transform(X_treated)
        X_control = imputer.transform(X_control)

    # ========== 步骤1: 计算倾向得分 ==========
    print("\n[步骤1] 计算倾向得分...")

    X = np.vstack([X_treated, X_control])
    y = np.hstack([np.ones(n_treated), np.zeros(n_control)])

    # 使用Logistic回归计算倾向得分
    try:
        logit_model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
        logit_model.fit(X, y)
        pscores = logit_model.predict_proba(X)[:, 1]
    except Exception as e:
        print(f"Logit模型拟合失败: {e}，使用简化方法")
        # 如果Logit失败，使用简单的线性组合
        pscores = y  # 临时使用实际标签

    pscores_treated = pscores[:n_treated]
    pscores_control = pscores[n_treated:]

    print(f"处理组倾向得分: 均值={pscores_treated.mean():.4f}, 标准差={pscores_treated.std():.4f}")
    print(f"对照组倾向得分: 均值={pscores_control.mean():.4f}, 标准差={pscores_control.std():.4f}")

    # ========== 步骤2: 执行近邻匹配 (1:2 with replacement & caliper) ==========
    print("\n[步骤2] 执行改进的近邻匹配...")
    print("  - 有放回匹配 (With Replacement)")
    print("  - 1:2匹配 (每个处理组匹配2个对照组)")
    print("  - 卡尺限制 (Caliper = 0.05)")
    print("  - 使用NearestNeighbors算法")

    # 设置卡尺（caliper）- 只有倾向得分差异小于此值才允许匹配
    CALIPER = 0.05

    # 使用NearestNeighbors算法进行有放回匹配
    # 重塑倾向得分为二维数组
    pscores_control_2d = pscores_control.reshape(-1, 1)
    pscores_treated_2d = pscores_treated.reshape(-1, 1)

    # 使用NearestNeighbors找到最近的邻居
    # n_neighbors=2 表示1:2匹配
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree')
    nbrs.fit(pscores_control_2d)

    # 找到每个处理组样本的2个最近邻居
    distances, indices = nbrs.kneighbors(pscores_treated_2d)

    # 应用卡尺限制
    matched_pairs = []
    discarded_count = 0

    for i in range(len(pscores_treated)):
        # 检查该处理组的2个候选对照组是否都在卡尺范围内
        valid_matches = 0
        for j in range(2):  # 2个邻居
            dist = distances[i][j]
            idx = indices[i][j]
            if dist <= CALIPER:
                matched_pairs.append({
                    'treated_idx': i,
                    'control_idx': idx,
                    'treated_pscore': pscores_treated[i],
                    'control_pscore': pscores_control[idx],
                    'distance': dist
                })
                valid_matches += 1

        # 如果2个候选都不满足卡尺要求，则丢弃该处理组样本
        if valid_matches == 0:
            discarded_count += 1

    n_matched_units = len([p['treated_idx'] for p in matched_pairs])
    n_unique_treated = len(set([p['treated_idx'] for p in matched_pairs]))
    match_rate = n_unique_treated / n_treated * 100

    print(f"\n匹配结果统计:")
    print(f"  处理组总数: {n_treated}")
    print(f"  成功匹配的处理组: {n_unique_treated}")
    print(f"  匹配产生的对照组样本数: {len(matched_pairs)}")
    print(f"  因超出卡尺被丢弃: {discarded_count}")
    print(f"  匹配成功率: {match_rate:.2f}%")

    if n_unique_treated == 0:
        print(f"警告：年份 {year} 没有成功匹配的样本，跳过该年")
        continue

    # 提取匹配后的数据
    matched_treated_indices = [p['treated_idx'] for p in matched_pairs]
    matched_control_indices = [p['control_idx'] for p in matched_pairs]

    treated_matched = treated.iloc[matched_treated_indices].copy()
    control_matched = control.iloc[matched_control_indices].copy()

    # 计算权重（有放回匹配：一个对照组可能匹配多个处理组）
    # 统计每个对照组样本被使用的次数
    from collections import Counter
    control_usage = Counter(matched_control_indices)

    # 为处理组赋予权重（总是1.0）
    treated_matched['weight'] = 1.0

    # 为对照组赋予权重（1 / 被使用次数）
    control_matched['weight'] = control_matched.index.map(
        lambda idx: 1.0 / control_usage.get(control.index.get_loc(idx), 1)
    )

    print(f"\n权重分配统计:")
    print(f"  对照组唯一样本数: {len(control_usage)}")
    print(f"  对照组重复使用情况:")
    # 显示重复使用的对照组样本
    usage_counts = Counter(control_usage.values())
    for usage, count in sorted(usage_counts.items()):
        print(f"    被使用{usage}次: {count}个样本")

    # ========== 步骤3: 平衡性检验 ==========
    print("\n[步骤3] 平衡性检验...")

    balance_stats = []

    for var in covariates:
        # 处理组均值和标准差
        t_mean = treated_matched[var].mean()
        t_std = treated_matched[var].std()

        # 对照组均值和标准差
        c_mean = control_matched[var].mean()
        c_std = control_matched[var].std()

        # 计算标准化偏差
        # Standardized Bias = (mean_t - mean_c) / sqrt((sd_t^2 + sd_c^2) / 2)
        pooled_std = np.sqrt((t_std**2 + c_std**2) / 2)
        if pooled_std > 0:
            std_bias = abs((t_mean - c_mean) / pooled_std) * 100
        else:
            std_bias = 0

        # t检验
        t_stat, p_value = stats.ttest_ind(
            treated_matched[var].dropna(),
            control_matched[var].dropna(),
            equal_var=False
        )

        balance_stats.append({
            'variable': var,
            'treated_mean': t_mean,
            'control_mean': c_mean,
            'std_bias': std_bias,
            't_stat': t_stat,
            'p_value': p_value,
            'bias_reduced': std_bias < 10  # 是否小于10%
        })

    balance_df = pd.DataFrame(balance_stats)

    print(f"\n{'变量':<25} {'处理组均值':>12} {'对照组均值':>12} {'标准化偏差(%)':>15} {'P值':>10} {'是否平衡'}")
    print("-" * 90)
    for row in balance_df.itertuples():
        status = "[OK]" if row.bias_reduced else "[FAIL]"
        print(f"{row.variable:<25} {row.treated_mean:>12.4f} {row.control_mean:>12.4f} {row.std_bias:>15.2f} {row.p_value:>10.4f} {status}")

    # 汇总统计
    n_balanced = balance_df['bias_reduced'].sum()
    print(f"\n平衡变量数: {n_balanced} / {len(covariates)}")

    # 存储结果
    for _, row in treated_matched.iterrows():
        matched_results.append(row.to_dict())

    for _, row in control_matched.iterrows():
        matched_results.append(row.to_dict())

    # 存储平衡性检验结果
    balance_df['year'] = year
    balance_test_results.append(balance_df)

# ========== 步骤4: 整合所有年份的匹配结果 ==========
print(f"\n{'=' * 80}")
print("整合历年匹配结果")
print(f"{'=' * 80}")

matched_df = pd.DataFrame(matched_results)

print(f"\n匹配后总样本数: {len(matched_df)}")
print(f"匹配后处理组样本数: {(matched_df['treat'] == 1).sum()}")
print(f"匹配后对照组样本数: {(matched_df['treat'] == 0).sum()}")

# 检查每年的样本数
print(f"\n各年份匹配样本数:")
yearly_counts = matched_df.groupby(['year', 'treat']).size().unstack(fill_value=0)
print(yearly_counts)

# 保存匹配后的数据
output_file = 'CEADs_PSM_matched_data.xlsx'
matched_df.to_excel(output_file, index=False)
print(f"\n匹配后数据已保存: {output_file}")

# ========== 步骤5: 生成平衡性检验汇总报告 ==========
balance_summary = pd.concat(balance_test_results, ignore_index=True)

print(f"\n{'=' * 80}")
print("平衡性检验汇总报告")
print(f"{'=' * 80}")

# 按变量汇总
print("\n各变量在不同年份的标准化偏差:")
pivot_bias = balance_summary.pivot(index='year', columns='variable', values='std_bias')
print(pivot_bias)

print("\n各变量在不同年份的P值:")
pivot_pvalue = balance_summary.pivot(index='year', columns='variable', values='p_value')
print(pivot_pvalue)

# 计算平均标准化偏差
avg_bias = balance_summary.groupby('variable')['std_bias'].mean()
print("\n平均标准化偏差:")
print(avg_bias)

# 检验是否满足平衡性假设
n_pass = (avg_bias < 10).sum()
print(f"\n满足平衡性假设的变量数: {n_pass} / {len(covariates)}")

# 保存平衡性检验结果
balance_summary.to_excel('PSM_balance_test_results.xlsx', index=False)
print(f"\n平衡性检验结果已保存: PSM_balance_test_results.xlsx")

# ========== 步骤6: 可视化 ==========
print(f"\n{'=' * 80}")
print("生成可视化图表")
print(f"{'=' * 80}")

# 图1: 倾向得分分布对比
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 选择几个代表性年份进行可视化
sample_years = sorted(matched_df['year'].unique())[::len(years)//4][:4]

for idx, year in enumerate(sample_years):
    ax = axes[idx // 2, idx % 2]

    year_data = matched_df[matched_df['year'] == year]

    # 计算该年的倾向得分（重新计算以便可视化）
    # 这里简化处理，使用实际的处理组标识作为倾向得分的代理
    treated_scores = year_data[year_data['treat'] == 1]['ln_pgdp'].values
    control_scores = year_data[year_data['treat'] == 0]['ln_pgdp'].values

    ax.hist(treated_scores, bins=10, alpha=0.5, label='处理组', color='red', density=True)
    ax.hist(control_scores, bins=10, alpha=0.5, label='对照组', color='blue', density=True)
    ax.set_xlabel('ln_pgdp (示例变量)')
    ax.set_ylabel('密度')
    ax.set_title(f'年份 {year} - 协变量分布对比')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('PSM_covariate_distribution.png', dpi=300, bbox_inches='tight')
print("协变量分布对比图已保存: PSM_covariate_distribution.png")

# 图2: 标准化偏差趋势图
fig, ax = plt.subplots(figsize=(12, 6))

for var in covariates:
    var_data = balance_summary[balance_summary['variable'] == var]
    ax.plot(var_data['year'], var_data['std_bias'], 'o-', label=var, linewidth=2, markersize=6)

ax.axhline(y=10, color='red', linestyle='--', linewidth=1.5, label='10%阈值')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
ax.set_xlabel('年份', fontsize=12)
ax.set_ylabel('标准化偏差 (%)', fontsize=12)
ax.set_title('平衡性检验：各变量标准化偏差趋势', fontsize=14, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('PSM_standardized_bias_trend.png', dpi=300, bbox_inches='tight')
print("标准化偏差趋势图已保存: PSM_standardized_bias_trend.png")

print(f"\n{'=' * 80}")
print("PSM分析完成！")
print(f"{'=' * 80}")
print(f"\n输出文件:")
print(f"1. {output_file} - 匹配后的面板数据")
print(f"2. PSM_balance_test_results.xlsx - 平衡性检验详细结果")
print(f"3. PSM_covariate_distribution.png - 协变量分布对比图")
print(f"4. PSM_standardized_bias_trend.png - 标准化偏差趋势图")
