# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import rcParams
from collections import Counter

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("倾向得分匹配（PSM）分析 - 使用缩尾后数据")
print("=" * 80)

# 读取缩尾后的数据
df = pd.read_excel('CEADs_最终数据集_2007-2019_V2_插值版_带DID_treat_缩尾版.xlsx')

# 定义匹配变量（协变量）
covariates = ['ln_pgdp', 'ln_pop_density', 'industrial_advanced', 'ln_fdi_openness', 'financial_development']

print(f"\n匹配变量（协变量）:")
for i, var in enumerate(covariates, 1):
    print(f"{i}. {var}")

# 获取年份列表
years = sorted(df['year'].unique())
print(f"\n年份范围: {years[0]} - {years[-1]} (共{len(years)}年)")

# 初始化存储所有年份匹配后的数据
all_matched_data = []

# 按年份进行PSM匹配
for year in years:
    print("\n" + "=" * 80)
    print(f"年份: {year}")
    print("=" * 80)

    # 提取当年数据
    df_year = df[df['year'] == year].copy()

    # 分离处理组和对照组
    treated = df_year[df_year['treat'] == 1].copy()
    control = df_year[df_year['treat'] == 0].copy()

    n_treated = len(treated)
    n_control = len(control)

    print(f"\n处理组样本数: {n_treated}")
    print(f"对照组样本数: {n_control}")

    if n_treated == 0 or n_control == 0:
        print(f"警告：年份 {year} 处理组或对照组为空，跳过该年")
        continue

    # ========== 步骤1: 估计倾向得分 ==========
    print("\n[步骤1] 估计倾向得分...")

    X = control[covariates].values
    y = control['treat'].values

    # 如果样本量足够，使用Logistic回归
    if len(X) > 10:
        try:
            log_model = LogisticRegression(solver='lbfgs', max_iter=1000)
            log_model.fit(X, y)
            # 为对照组计算倾向得分
            pscores_control = log_model.predict_proba(X)[:, 1]
            # 为处理组计算倾向得分
            pscores_treated = log_model.predict_proba(treated[covariates].values)[:, 1]
        except:
            pscores_control = np.zeros(len(control))
            pscores_treated = np.ones(len(treated))
    else:
        pscores_control = np.zeros(len(control))
        pscores_treated = np.ones(len(treated))

    print(f"处理组倾向得分: 均值={pscores_treated.mean():.4f}, 标准差={pscores_treated.std():.4f}")
    print(f"对照组倾向得分: 均值={pscores_control.mean():.4f}, 标准差={pscores_control.std():.4f}")

    # ========== 步骤2: 执行近邻匹配 (1:2 with replacement & caliper) ==========
    print("\n[步骤2] 执行改进的近邻匹配...")
    print("  - 有放回匹配 (With Replacement)")
    print("  - 1:2匹配 (每个处理组匹配2个对照组)")
    print("  - 卡尺限制 (Caliper = 0.05)")
    print("  - 使用NearestNeighbors算法")

    CALIPER = 0.05

    # 使用NearestNeighbors算法进行有放回匹配
    pscores_control_2d = pscores_control.reshape(-1, 1)
    pscores_treated_2d = pscores_treated.reshape(-1, 1)

    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree')
    nbrs.fit(pscores_control_2d)

    distances, indices = nbrs.kneighbors(pscores_treated_2d)

    # 应用卡尺限制
    matched_pairs = []
    discarded_count = 0

    for i in range(len(pscores_treated)):
        valid_matches = 0
        for j in range(2):
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

    # 计算权重
    control_usage = Counter(matched_control_indices)

    treated_matched['weight'] = 1.0
    control_matched['weight'] = control_matched.index.map(
        lambda idx: 1.0 / control_usage.get(control.index.get_loc(idx), 1)
    )

    # 合并处理组和对照组
    year_matched = pd.concat([treated_matched, control_matched], ignore_index=True)
    all_matched_data.append(year_matched)

# 合并所有年份的匹配数据
df_matched = pd.concat(all_matched_data, ignore_index=True)

print(f"\n{'=' * 80}")
print(f"PSM匹配完成！")
print(f"{'=' * 80}")
print(f"匹配后总样本数: {len(df_matched)}")

# 保存匹配后的数据
output_file = 'CEADs_PSM_matched_data_winsorized.xlsx'
df_matched.to_excel(output_file, index=False)

print(f"\n匹配后的数据已保存: {output_file}")

# 进行去重处理
print(f"\n{'=' * 80}")
print("进行去重处理...")
print(f"{'=' * 80}")

before_dedup = len(df_matched)
df_dedup = df_matched.drop_duplicates(subset=['city_name', 'year'], keep='first')
after_dedup = len(df_dedup)

print(f"去重前样本数: {before_dedup}")
print(f"去重后样本数: {after_dedup}")
print(f"删除的重复行数: {before_dedup - after_dedup}")

# 保存去重后的数据
output_file_dedup = 'CEADs_PSM_matched_data_winsorized_dedup.xlsx'
df_dedup.to_excel(output_file_dedup, index=False)

print(f"\n去重后的数据已保存: {output_file_dedup}")

print(f"\n{'=' * 80}")
print("PSM分析完成！")
print(f"{'=' * 80}")
