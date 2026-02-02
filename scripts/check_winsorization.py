# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

print("=" * 80)
print("检查 financial_development 是否进行过缩尾处理")
print("=" * 80)

# 读取原始数据集（PSM匹配前）
print("\n读取原始数据集...")
df_original = pd.read_excel('CEADs_最终数据集_2007-2019_V2_插值版_带DID_treat.xlsx')

# 读取PSM匹配并去重后的数据集
print("读取PSM匹配后数据集...")
df_psm = pd.read_excel('CEADs_PSM_matched_data_dedup.xlsx')

print(f"\n原始数据集样本数: {len(df_original)}")
print(f"PSM匹配后样本数: {len(df_psm)}")

# 检查financial_development变量
var = 'financial_development'

print("\n" + "=" * 80)
print(f"变量: {var}")
print("=" * 80)

print("\n原始数据集统计:")
print(f"  样本数: {df_original[var].notna().sum()}")
print(f"  均值: {df_original[var].mean():.6f}")
print(f"  标准差: {df_original[var].std():.6f}")
print(f"  最小值: {df_original[var].min():.6f}")
print(f"  最大值: {df_original[var].max():.6f}")
print(f"  中位数: {df_original[var].median():.6f}")
print(f"  1% 分位数: {df_original[var].quantile(0.01):.6f}")
print(f"  99% 分位数: {df_original[var].quantile(0.99):.6f}")

print("\nPSM匹配后统计:")
print(f"  样本数: {df_psm[var].notna().sum()}")
print(f"  均值: {df_psm[var].mean():.6f}")
print(f"  标准差: {df_psm[var].std():.6f}")
print(f"  最小值: {df_psm[var].min():.6f}")
print(f"  最大值: {df_psm[var].max():.6f}")
print(f"  中位数: {df_psm[var].median():.6f}")
print(f"  1% 分位数: {df_psm[var].quantile(0.01):.6f}")
print(f"  99% 分位数: {df_psm[var].quantile(0.99):.6f}")

# 检查最大值和最小值是否相同
print("\n" + "=" * 80)
print("对比分析:")
print("=" * 80)

if df_original[var].max() == df_psm[var].max():
    print(f"[相同] 最大值: {df_original[var].max():.6f}")
else:
    print(f"[不同] 原始最大值: {df_original[var].max():.6f}, PSM后: {df_psm[var].max():.6f}")

if df_original[var].min() == df_psm[var].min():
    print(f"[相同] 最小值: {df_original[var].min():.6f}")
else:
    print(f"[不同] 原始最小值: {df_original[var].min():.6f}, PSM后: {df_psm[var].min():.6f}")

# 检查是否被缩尾处理
print("\n" + "=" * 80)
print("缩尾处理判断:")
print("=" * 80)

# 检查1%和99%分位数之外是否有值
p01_orig = df_original[var].quantile(0.01)
p99_orig = df_original[var].quantile(0.99)
p01_psm = df_psm[var].quantile(0.01)
p99_psm = df_psm[var].quantile(0.99)

print(f"\n原始数据集:")
print(f"  超过99%分位数({p99_orig:.4f})的样本数: {(df_original[var] > p99_orig).sum()}")
print(f"  低于1%分位数({p01_orig:.4f})的样本数: {(df_original[var] < p01_orig).sum()}")

print(f"\nPSM匹配后数据集:")
print(f"  超过99%分位数({p99_psm:.4f})的样本数: {(df_psm[var] > p99_psm).sum()}")
print(f"  低于1%分位数({p01_psm:.4f})的样本数: {(df_psm[var] < p01_psm).sum()}")

# 检查极端值案例
print("\n" + "=" * 80)
print("极端值案例对比:")
print("=" * 80)

print("\n原始数据集 - 最高的5个值:")
top5_orig = df_original.nlargest(5, var)[['city_name', 'year', var]]
print(top5_orig.to_string(index=False))

print("\nPSM匹配后 - 最高的5个值:")
top5_psm = df_psm.nlargest(5, var)[['city_name', 'year', var]]
print(top5_psm.to_string(index=False))

# 检查是否有明显的截断迹象
print("\n" + "=" * 80)
print("结论:")
print("=" * 80)

# 如果最大值大于99%分位数很多，可能没有缩尾
if df_psm[var].max() > p99_psm * 1.1:
    print("[判断] financial_development 似乎没有进行过缩尾处理")
    print(f"  原因: 最大值({df_psm[var].max():.4f}) 远大于99%分位数({p99_psm:.4f})")
elif df_psm[var].max() == p99_psm:
    print("[判断] financial_development 可能进行了1%缩尾处理")
    print(f"  原因: 最大值({df_psm[var].max():.4f}) 等于99%分位数({p99_psm:.4f})")
else:
    print("[判断] 无法明确判断是否进行了缩尾处理")
    print(f"  最大值: {df_psm[var].max():.4f}, 99%分位数: {p99_psm:.4f}")

print("\n建议:")
print("  - 如果存在极端值影响回归结果，可以考虑进行1%或5%的缩尾处理")
print("  - 常见方法：将超过99%分位数的值替换为99%分位数")
print("  - 将低于1%分位数的值替换为1%分位数")
