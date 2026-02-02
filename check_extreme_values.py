# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("极端值检查：ln_fdi_openness 和 financial_development")
print("=" * 80)

# 读取数据
df = pd.read_excel('CEADs_PSM_matched_data_dedup.xlsx')

print(f"\n数据集样本数: {len(df)}")
print(f"数据集列数: {len(df.columns)}")

# 检查的变量
vars_to_check = ['ln_fdi_openness', 'financial_development']

for var in vars_to_check:
    print("\n" + "=" * 80)
    print(f"变量: {var}")
    print("=" * 80)

    # 基本统计
    print(f"\n基本统计量:")
    print(f"  样本数: {df[var].notna().sum()}")
    print(f"  缺失值: {df[var].isna().sum()}")
    print(f"  均值: {df[var].mean():.6f}")
    print(f"  中位数: {df[var].median():.6f}")
    print(f"  标准差: {df[var].std():.6f}")
    print(f"  最小值: {df[var].min():.6f}")
    print(f"  最大值: {df[var].max():.6f}")
    print(f"  极差: {df[var].max() - df[var].min():.6f}")

    # 分位数
    print(f"\n分位数:")
    print(f"  1% 分位数: {df[var].quantile(0.01):.6f}")
    print(f"  5% 分位数: {df[var].quantile(0.05):.6f}")
    print(f"  25% 分位数: {df[var].quantile(0.25):.6f}")
    print(f"  50% 分位数: {df[var].quantile(0.50):.6f}")
    print(f"  75% 分位数: {df[var].quantile(0.75):.6f}")
    print(f"  95% 分位数: {df[var].quantile(0.95):.6f}")
    print(f"  99% 分位数: {df[var].quantile(0.99):.6f}")

    # 使用IQR方法检测异常值
    Q1 = df[var].quantile(0.25)
    Q3 = df[var].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_lower = df[df[var] < lower_bound]
    outliers_upper = df[df[var] > upper_bound]
    n_outliers = len(outliers_lower) + len(outliers_upper)

    print(f"\nIQR方法异常值检测:")
    print(f"  Q1 (25%): {Q1:.6f}")
    print(f"  Q3 (75%): {Q3:.6f}")
    print(f"  IQR: {IQR:.6f}")
    print(f"  下界 (Q1 - 1.5*IQR): {lower_bound:.6f}")
    print(f"  上界 (Q3 + 1.5*IQR): {upper_bound:.6f}")
    print(f"  异常值数量: {n_outliers} ({n_outliers/len(df)*100:.2f}%)")
    print(f"  极端低值: {len(outliers_lower)}")
    print(f"  极端高值: {len(outliers_upper)}")

    # 使用3σ方法检测异常值
    mean = df[var].mean()
    std = df[var].std()
    outliers_3sigma_lower = df[df[var] < mean - 3*std]
    outliers_3sigma_upper = df[df[var] > mean + 3*std]
    n_outliers_3sigma = len(outliers_3sigma_lower) + len(outliers_3sigma_upper)

    print(f"\n3σ方法异常值检测:")
    print(f"  均值: {mean:.6f}")
    print(f"  标准差: {std:.6f}")
    print(f"  下界 (μ - 3σ): {mean - 3*std:.6f}")
    print(f"  上界 (μ + 3σ): {mean + 3*std:.6f}")
    print(f"  异常值数量: {n_outliers_3sigma} ({n_outliers_3sigma/len(df)*100:.2f}%)")
    print(f"  极端低值: {len(outliers_3sigma_lower)}")
    print(f"  极端高值: {len(outliers_3sigma_upper)}")

    # 显示极端值案例（前5个和后5个）
    print(f"\n极端值案例（最高值的前5个）:")
    top5 = df.nlargest(5, var)[['city_name', 'year', var]]
    print(top5.to_string(index=False))

    print(f"\n极端值案例（最低值的前5个）:")
    bottom5 = df.nsmallest(5, var)[['city_name', 'year', var]]
    print(bottom5.to_string(index=False))

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, var in enumerate(vars_to_check):
    # 箱线图
    ax1 = axes[idx, 0]
    ax1.boxplot(df[var].dropna(), vert=True)
    ax1.set_title(f'{var} - 箱线图', fontsize=12, fontweight='bold')
    ax1.set_ylabel(var, fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 直方图
    ax2 = axes[idx, 1]
    ax2.hist(df[var].dropna(), bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(df[var].mean(), color='red', linestyle='--', linewidth=2, label=f'均值: {df[var].mean():.2f}')
    ax2.axvline(df[var].median(), color='green', linestyle='--', linewidth=2, label=f'中位数: {df[var].median():.2f}')
    ax2.set_title(f'{var} - 分布直方图', fontsize=12, fontweight='bold')
    ax2.set_xlabel(var, fontsize=10)
    ax2.set_ylabel('频数', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('extreme_values_check.png', dpi=300, bbox_inches='tight')
print("\n" + "=" * 80)
print("可视化已保存: extreme_values_check.png")
print("=" * 80)

# 处理组与对照组的对比
print("\n" + "=" * 80)
print("处理组 vs 对照组对比")
print("=" * 80)

for var in vars_to_check:
    print(f"\n变量: {var}")
    print("-" * 80)

    treated = df[df['treat'] == 1][var].dropna()
    control = df[df['treat'] == 0][var].dropna()

    print(f"{'统计量':<20} {'处理组':<15} {'对照组':<15} {'差异'}")
    print("-" * 80)
    print(f"{'样本数':<20} {len(treated):<15} {len(control):<15}")
    print(f"{'均值':<20} {treated.mean():<15.6f} {control.mean():<15.6f} {treated.mean()-control.mean():.6f}")
    print(f"{'标准差':<20} {treated.std():<15.6f} {control.std():<15.6f}")
    print(f"{'最小值':<20} {treated.min():<15.6f} {control.min():<15.6f}")
    print(f"{'最大值':<20} {treated.max():<15.6f} {control.max():<15.6f}")

print("\n" + "=" * 80)
print("极端值检查完成！")
print("=" * 80)
