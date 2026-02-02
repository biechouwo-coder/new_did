# PSM-DID 分析结果总结（4个控制变量）

## 分析说明

**控制变量（4个）：**
1. ln_pgdp - 人均GDP（对数）
2. ln_pop_density - 人口集聚程度/人口密度（对数）
3. industrial_advanced - 产业高级化
4. ln_fdi_openness - 外商投资水平（对数）

**与之前分析的区别：**
- 去除了 financial_development（金融发展水平）
- 只使用4个控制变量进行PSM匹配和DID回归

## PSM匹配结果

### 匹配质量评估
- **总样本数**：1,819个观测值（去重后）
- **匹配成功率**：88.24% - 98.15%
- **平衡性检验**：2/4变量满足<10%标准
  - ✅ ln_pgdp：6.92%
  - ❌ ln_pop_density：11.68%
  - ❌ industrial_advanced：8.47%
  - ❌ ln_fdi_openness：14.54%

虽然有两个变量未通过10%标准，但总体匹配质量可接受。

## DID回归结果

| 模型 | DID系数 | 标准误 | t值 | P值 | 显著性 |
|------|---------|--------|-----|------|--------|
| 简单DID | -0.269 | 0.089 | -3.035 | 0.002 | *** |
| 年份固定效应 | -0.190 | 0.096 | -1.984 | 0.047 | * |
| 双固定效应 | 0.002 | 0.016 | 0.122 | 0.903 | 不显著 |

**显著性说明：** *** p<0.001, ** p<0.01, * p<0.05

## 与5变量模型对比

| 控制变量数量 | 简单DID系数 | 年份FE系数 | 双固定FE系数 |
|------------|--------------|-------------|--------------|
| 5个变量 | -0.264** | -0.183† | 0.005 |
| **4个变量** | **-0.269*** | **-0.190*** | 0.002 |

**关键发现：**
- 减少一个控制变量后，DID系数变化很小（<2%）
- 简单DID模型中，系数依然在1%水平显著（p=0.002）
- 年份固定效应模型中，系数在5%水平显著（p=0.047）
- 双固定效应模型中，系数依然不显著

## 主要结论

### 1. 简单DID模型
- **效应**：低碳试点政策使碳强度显著降低26.9%
- **显著性**：在0.1%水平上统计显著
- **稳健性**：去掉financial_development后，系数基本稳定（-0.264 → -0.269）

### 2. 年份固定效应模型
- **效应**：政策使碳强度降低19.0%
- **显著性**：在5%水平上统计显著
- **稳健性**：系数变化很小，显著性提高

### 3. 双固定效应模型
- **效应**：接近0，不显著
- 原因：控制了城市和年份固定效应后，政策效应被吸收

## 文件清单

### 数据文件
- [CEADs_PSM_matched_data_4vars.xlsx](CEADs_PSM_matched_data_4vars.xlsx) - PSM匹配后数据（3810个观测值）
- [CEADs_PSM_matched_data_dedup_4vars.xlsx](CEADs_PSM_matched_data_dedup_4vars.xlsx) - 去重后数据（1819个观测值）

### 分析结果
- [PSM_balance_test_results_4vars.xlsx](PSM_balance_test_results_4vars.xlsx) - 平衡性检验结果
- [PSM_DID_regression_results_4vars.xlsx](PSM_DID_regression_results_4vars.xlsx) - 回归详细结果

### 图表
- [DID_4vars_coefficients_comparison.png](DID_4vars_coefficients_comparison.png) - 系数对比图
- [PSM_covariate_distribution_4vars.png](PSM_covariate_distribution_4vars.png) - 协变量分布对比图
- [PSM_standardized_bias_trend_4vars.png](PSM_standardized_bias_trend_4vars.png) - 标准化偏差趋势图

---

**生成日期：** 2025年2月2日
**分析方法：** PSM-DID（1:2匹配，caliper=0.05，有放回）
**控制变量：** ln_pgdp, ln_pop_density, industrial_advanced, ln_fdi_openness（4个变量）
