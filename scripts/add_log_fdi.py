# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

# 读取主数据集
df = pd.read_excel('CEADs_最终数据集_2007-2019_V2_插值版_带DID_treat.xlsx')

print("=" * 80)
print("添加 fdi_openness 对数变量")
print("=" * 80)

print(f"\n原始数据集样本数: {len(df)}")
print(f"原始列数: {len(df.columns)}")

# 检查fdi_openness变量
if 'fdi_openness' in df.columns:
    print(f"\nfdi_openness 变量存在")
    print(f"  均值: {df['fdi_openness'].mean():.4f}")
    print(f"  标准差: {df['fdi_openness'].std():.4f}")
    print(f"  最小值: {df['fdi_openness'].min():.4f}")
    print(f"  最大值: {df['fdi_openness'].max():.4f}")
    print(f"  缺失值数: {df['fdi_openness'].isna().sum()}")

    # 检查是否有负值或零值
    n_zero_or_negative = (df['fdi_openness'] <= 0).sum()
    print(f"  <= 0的值: {n_zero_or_negative}")

    # 计算对数（处理零值和负值）
    # 方法：如果值 <= 0，设为一个很小的正数或者用NaN
    df['ln_fdi_openness'] = np.where(
        df['fdi_openness'] > 0,
        np.log(df['fdi_openness']),
        np.nan  # 对 <= 0 的值设为NaN
    )

    print(f"\n生成的 ln_fdi_openness 变量:")
    print(f"  均值: {df['ln_fdi_openness'].mean():.4f}")
    print(f"  标准差: {df['ln_fdi_openness'].std():.4f}")
    print(f"  最小值: {df['ln_fdi_openness'].min():.4f}")
    print(f"  最大值: {df['ln_fdi_openness'].max():.4f}")
    print(f"  缺失值数: {df['ln_fdi_openness'].isna().sum()}")

    # 显示数据示例
    print(f"\n数据示例（前10行）:")
    print(df[['city_name', 'year', 'fdi_openness', 'ln_fdi_openness']].head(10).to_string(index=False))

else:
    print(f"\n警告：数据集中不存在 fdi_openness 变量")

# 保存更新后的数据集
output_file = 'CEADs_最终数据集_2007-2019_V2_插值版_带DID_treat.xlsx'
df.to_excel(output_file, index=False)

print(f"\n{'=' * 80}")
print(f"数据集已更新并保存: {output_file}")
print(f"总列数: {len(df.columns)}")
print(f"{'=' * 80}")

print(f"\n所有列名:")
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col}")
