# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

print("=" * 80)
print("生成滞后 DID 变量 (DID_lag1)")
print("定义：政策实施后一年才标记为1")
print("=" * 80)

# 读取缩尾后的原始数据
df = pd.read_excel('CEADs_最终数据集_2007-2019_V2_插值版_带DID_treat_缩尾版.xlsx')

print(f"\n原始数据样本数: {len(df)}")
print(f"年份范围: {df['year'].min()} - {df['year'].max()}")

# 从DID变量推断每个城市的政策开始年份
# 对于每个城市，找到DID第一次变为1的年份
city_policy_years = {}

for city in df['city_name'].unique():
    city_data = df[df['city_name'] == city].sort_values('year')
    # 找到第一个DID=1的年份
    did_1_years = city_data[city_data['DID'] == 1]['year']
    if len(did_1_years) > 0:
        city_policy_years[city] = did_1_years.min()

print(f"\n从DID变量推断出{len(city_policy_years)}个城市的政策开始年份")

# 生成DID_lag1变量
def get_did_lag1(row):
    city = row['city_name']
    year = row['year']

    if city in city_policy_years:
        policy_year = city_policy_years[city]
        # 政策实施后一年才标记为1
        if year >= policy_year + 1:
            return 1
    return 0

df['DID_lag1'] = df.apply(get_did_lag1, axis=1)

# 统计对比
print(f"\n原始DID变量统计:")
print(df['DID'].value_counts().sort_index())

print(f"\n滞后DID_lag1变量统计:")
print(df['DID_lag1'].value_counts().sort_index())

# 查看变化
changed = (df['DID'] != df['DID_lag1']).sum()
print(f"\nDID与DID_lag1不同的观测值数: {changed} ({changed/len(df)*100:.2f}%)")

# 查看具体案例
print(f"\n具体案例（前10个变化的观测值）:")
changes_df = df[df['DID'] != df['DID_lag1']][['city_name', 'year', 'DID', 'DID_lag1']].head(10)
print(changes_df.to_string(index=False))

# 保存带有DID_lag1的数据
output_file = 'CEADs_最终数据集_2007-2019_V2_插值版_带DID_lag1_缩尾版.xlsx'
df.to_excel(output_file, index=False)

print(f"\n带有DID_lag1变量的数据已保存: {output_file}")

# 交叉表
print(f"\nDID × DID_lag1 交叉表:")
print(pd.crosstab(df['DID'], df['DID_lag1'], margins=True))

# 按年份统计对比
print(f"\n按年份统计DID和DID_lag1:")
yearly_comparison = df.groupby('year')[['DID', 'DID_lag1']].sum()
print(yearly_comparison)

print(f"\n{'=' * 80}")
print("DID_lag1变量生成完成！")
print(f"{'=' * 80}")
