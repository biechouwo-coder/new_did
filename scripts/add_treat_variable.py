# -*- coding: utf-8 -*-
import pandas as pd

# 读取数据
df = pd.read_excel('CEADs_最终数据集_2007-2019_V2_插值版_带DID.xlsx')

print("=" * 60)
print("生成分组变量 treat")
print("=" * 60)

print(f"\n原始数据样本数: {len(df)}")
print(f"城市数量: {df['city_name'].nunique()}")

# 查看DID列的统计
print(f"\nDID列统计:")
print(df['DID'].value_counts().sort_index())

# 找出每个城市是否有任何一年DID=1
city_treat_status = df.groupby('city_name')['DID'].max().reset_index()
city_treat_status.columns = ['city_name', 'treat']

print(f"\n实验组城市数量: {(city_treat_status['treat'] == 1).sum()}")
print(f"对照组城市数量: {(city_treat_status['treat'] == 0).sum()}")

# 列出实验组城市
treated_cities = city_treat_status[city_treat_status['treat'] == 1]['city_name'].tolist()
print(f"\n实验组城市（共{len(treated_cities)}个）:")
for i, city in enumerate(treated_cities, 1):
    print(f"{i}. {city}")

# 将treat变量合并回原数据
df = df.merge(city_treat_status, on='city_name', how='left')

# 验证：对于实验组城市，至少有一年DID=1
print(f"\n验证：实验组城市DID=1的年份数:")
for city in treated_cities[:5]:  # 只显示前5个城市作为示例
    city_data = df[df['city_name'] == city]
    did_years = city_data[city_data['DID'] == 1]['year'].tolist()
    print(f"  {city}: {did_years}")

if len(treated_cities) > 5:
    print(f"  ... (其余{len(treated_cities)-5}个城市)")

# 查看最终数据结构
print(f"\n最终数据列:")
print(list(df.columns))

print(f"\ntreat变量统计:")
print(df['treat'].value_counts())

# 检查交叉表：treat vs DID
print(f"\ntreat × DID 交叉表:")
print(pd.crosstab(df['treat'], df['DID'], margins=True))

# 保存结果
output_file = 'CEADs_最终数据集_2007-2019_V2_插值版_带DID_treat.xlsx'
df.to_excel(output_file, index=False)

print(f"\n处理完成！已保存到: {output_file}")

# 显示部分数据示例
print(f"\n数据示例（前10行）:")
print(df[['city_name', 'year', 'DID', 'treat']].head(10))
