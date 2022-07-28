# Binary Classification

## 简介

此任务主要是考察如何训练 classification 模型，如何衡量 classification 模型的好坏，还有一些模型的特征处理。

[google crash course 链接](https://developers.google.com/machine-learning/crash-course/classification/programming-exercise?hl=zh-cn)


## 环境

- python: 3.7.13
- pandas: 1.3.5
- numpy: 1.21.5
- tensorflow: 2.0.0
- matplotlib: 3.5.1

## 数据集

训练过程使用的数据集是 google 提供的 [California Housing Dataset](https://developers.google.com/machine-learning/crash-course/california-housing-data-description)

## 代码结构

### 1. 模块导入

``` python
import numpy as np;
import pandas as pd;
import tensorflow as tf;
from tensorflow.keras import layers;
from matplotlib import pyplot as plt;
```

其中：
- numpy：python 的扩展库，支持大量的维度数组与矩阵运算
- pandas: python 的扩展库，用于做各种数据分析操作。比如：数据集导入，数据的归并、清洗和加工等
- matplotlib: python 的绘图库


### 2. 设置展示的数据格式

``` python
# 设置数据展示格式
pd.options.display.max_rows = 10;
pd.options.display.float_format = "{:.1f}".format;
tf.keras.backend.set_floatx('float32');
```

### 3. 载入数据并打乱数据顺序使其分布均匀

```python
# 读取并打乱数据
train_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")
test_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv")
train_df = train_df.reindex(np.random.permutation(train_df.index)); # shuffle the training set
```

使用 reindex 方法实现 shuffle 效果。

### 4. normalize 数据

```python
train_df_mean = train_df.mean();
train_df_std = train_df.std();
train_df_norm = (train_df - train_df_mean)/train_df_std;
```

normalize 可以使每个属性的数据均分布在某个范围内（比如：-3～3），这样训练起来更快。如果不做 normalize 操作，模型在训练的时候就会过度关注数值范围
更大的属性上。

### 5. 自定义 label

```python
# create binary label
threshold = 265000;
train_df_norm["median_house_value_is_high"] = (train_df["median_house_value"] > threshold).astype(float);
test_df_norm["median_house_value_is_high"] = (test_df["median_house_value"] > threshold).astype(float);
train_df_norm["median_house_value_is_high"].head(8000)
```

训练过程使用的数据集比较特别，和常见的数据集不一样，它是没有 label 的，所以需要我们自己设置 cut point。大于 cut point 的数据划分为 1（即 
median house value is high），小于 cut point 的数据划分为 0（即 median house value is not high）。

这里我们基于 raw data(即 train_df) 做 cut point，也可以给予 normalize 后的 data(即 train_df_norm) 做 cut point：

```python
threshold_in_Z = 1.0 
train_df_norm["median_house_value_is_high"] = (train_df_norm["median_house_value"] > threshold_in_Z).astype(float)
test_df_norm["median_house_value_is_high"] = (test_df_norm["median_house_value"] > threshold_in_Z).astype(float) 
```

不同的数据集对应设置不同的 cut point。

### 6. 定义 feature columns

```python
feature_columns = [];

median_income = tf.feature_column.numeric_column("median_income");
feature_columns.append(median_income);

tr = tf.feature_column.numeric_column("total_rooms");
feature_columns.append(tr);

feature_layer = layers.DenseFeatures(feature_columns);

feature_layer(dict(train_df_norm));
```

