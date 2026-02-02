# -*- coding: utf-8 -*-
import pandas as pd

print("=" * 80)
print("PSM匹配后数据去重处理")
print("=" * 80)

# 读取PSM匹配后的数据
df = pd.read_excel('CEADs_PSM_matched_data.xlsx')

print(f"\n原始数据概况:")
print(f"总样本数: {len(df)}")
print(f"列数: {len(df.columns)}")
print(f"\n列名: {list(df.columns)}")

# 检查是否存在重复
print(f"\n重复数据检查:")

# 统计每个城市-年份组合的出现次数
city_year_counts = df.groupby(['city_name', 'year']).size()
duplicates = city_year_counts[city_year_counts > 1]

if len(duplicates) > 0:
    print(f"\n发现重复的城市-年份组合: {len(duplicates)}个")
    print(f"\n重复详情（前10个）:")
    dup_df = duplicates.head(10).reset_index()
    for idx, row in dup_df.iterrows():
        print(f"  城市: {row['city_name']}, 年份: {row['year']}, 出现次数: {row[0]}")

    if len(duplicates) > 10:
        print(f"  ... (还有{len(duplicates)-10}个重复组合)")
else:
    print("未发现重复数据，数据已是唯一的。")

# 去重处理：按照city_name和year去重，保留第一条记录
print(f"\n执行去重操作...")
print(f"去重策略: 按照 ['city_name', 'year'] 去重，保留第一次出现的记录")

# 去重前样本数
before_dedup = len(df)

# 执行去重
df_dedup = df.drop_duplicates(subset=['city_name', 'year'], keep='first')

# 去重后样本数
after_dedup = len(df_dedup)

# 统计去重结果
duplicates_removed = before_dedup - after_dedup

print(f"\n去重结果:")
print(f"  去重前样本数: {before_dedup}")
print(f"  去重后样本数: {after_dedup}")
print(f"  删除的重复行数: {duplicates_removed}")

# 检查去重后的数据结构
print(f"\n去重后数据分布:")
yearly_counts = df_dedup.groupby(['year', 'treat']).size().unstack(fill_value=0)
print(yearly_counts)

print(f"\n去重后总样本数: {after_dedup}")
print(f"去重后处理组样本数: {(df_dedup['treat'] == 1).sum()}")
print(f"去重后对照组样本数: {(df_dedup['treat'] == 0).sum()}")

# 验证去重是否成功
print(f"\n去重验证:")
final_city_year_counts = df_dedup.groupby(['city_name', 'year']).size()
if (final_city_year_counts > 1).any():
    print("[WARNING] 警告：去重后仍存在重复！")
else:
    print("[OK] 验证通过：每个城市每年只有一条记录")

# 保存去重后的数据
output_file = 'CEADs_PSM_matched_data_dedup.xlsx'
df_dedup.to_excel(output_file, index=False)

print(f"\n去重后的数据已保存: {output_file}")

# 对比去重前后的统计
print(f"\n去重前后对比:")
print(f"{'指标':<20} {'去重前':>12} {'去重后':>12} {'变化'}")
print("-" * 60)
print(f"{'总样本数':<20} {before_dedup:>12} {after_dedup:>12} {-duplicates_removed}")
print(f"{'处理组样本':<20} {(df['treat'] == 1).sum():>12} {(df_dedup['treat'] == 1).sum():>12} {-duplicates_removed}")
print(f"{'对照组样本':<20} {(df['treat'] == 0).sum():>12} {(df_dedup['treat'] == 0).sum():>12} {-duplicates_removed}")

print(f"\n{'=' * 80}")
print("去重处理完成！")
print(f"{'=' * 80}")
