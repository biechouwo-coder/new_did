# -*- coding: utf-8 -*-
import pandas as pd

df = pd.read_excel('CEADs_最终数据集_2007-2019_V2_插值版_带DID_treat_缩尾版.xlsx')

print('数据集列名:')
print('')
for i, col in enumerate(df.columns, 1):
    print(f'{i:2d}. {col}')

print('\n\n搜索包含"道路"或"road"的变量:')
found = False
for col in df.columns:
    if 'road' in col.lower() or '道路' in col:
        print(f'  - {col}')
        found = True
if not found:
    print('  未找到包含"道路"或"road"的变量')
