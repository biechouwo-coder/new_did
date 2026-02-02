# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

print("=" * 80)
print("对 financial_development 进行 1% 缩尾处理")
print("=" * 80)

# 读取原始数据
df = pd.read_excel('CEADs_最终数据集_2007-2019_V2_插值版_带DID_treat.xlsx')

print(f"\n原始数据样本数: {len(df)}")
print(f"原始数据列数: {len(df.columns)}")

# 显示缩尾前的统计
var = 'financial_development'
print(f"\n缩尾前 {var} 统计:")
print(f"  均值: {df[var].mean():.6f}")
print(f"  标准差: {df[var].std():.6f}")
print(f"  最小值: {df[var].min():.6f}")
print(f"  最大值: {df[var].max():.6f}")
print(f"  1% 分位数: {df[var].quantile(0.01):.6f}")
print(f"  99% 分位数: {df[var].quantile(0.99):.6f}")

# 计算1%和99%分位数
p01 = df[var].quantile(0.01)
p99 = df[var].quantile(0.99)

# 统计被缩尾的观测值
n_winsorized_low = (df[var] < p01).sum()
n_winsorized_high = (df[var] > p99).sum()

print(f"\n将被缩尾的观测值:")
print(f"  低于1%分位数的: {n_winsorized_low} 个")
print(f"  高于99%分位数的: {n_winsorized_high} 个")
print(f"  总计: {n_winsorized_low + n_winsorized_high} 个 ({(n_winsorized_low + n_winsorized_high)/len(df)*100:.2f}%)")

# 保存原始值用于展示
df_original_values = df[[var, 'city_name', 'year']].copy()

# 进行1%缩尾处理
df[var] = df[var].clip(lower=p01, upper=p99)

print(f"\n缩尾后 {var} 统计:")
print(f"  均值: {df[var].mean():.6f}")
print(f"  标准差: {df[var].std():.6f}")
print(f"  最小值: {df[var].min():.6f}")
print(f"  最大值: {df[var].max():.6f}")
print(f"  中位数: {df[var].median():.6f}")

# 保存缩尾后的数据
output_file = 'CEADs_最终数据集_2007-2019_V2_插值版_带DID_treat_缩尾版.xlsx'
df.to_excel(output_file, index=False)

print(f"\n缩尾后的数据已保存: {output_file}")

# 显示被缩尾的极端值案例
print("\n" + "=" * 80)
print("被缩尾的极端值案例:")
print("=" * 80)

extreme_low_indices = df_original_values[df_original_values[var] < p01].index
extreme_high_indices = df_original_values[df_original_values[var] > p99].index

print(f"\n被缩尾到1%分位数的案例（前5个）:")
if len(extreme_low_indices) > 0:
    extreme_low_cases = df_original_values.loc[extreme_low_indices][['city_name', 'year', var]].head(5)
    print(extreme_low_cases.to_string(index=False))
else:
    print("无")

print(f"\n被缩尾到99%分位数的案例（前5个）:")
if len(extreme_high_indices) > 0:
    extreme_high_cases = df_original_values.loc[extreme_high_indices][['city_name', 'year', var]].head(5)
    print(extreme_high_cases.to_string(index=False))
else:
    print("无")

print(f"\n{'=' * 80}")
print("缩尾处理完成！")
print(f"{'=' * 80}")
