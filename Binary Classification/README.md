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

normalize 可以使每个属性的数据均分布在某个范围内（比如：-3～3），这样训练起来更快。如果不做 normalize 操作，模型在训练的时候就会过度关注数值范围更大的属性上。

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

feature columns 的作用主要是做一些特征工程的工作。google 官方文档对它的定义是：

**feature columns-a data structure describing the features that an Estimator requires for training and inference**

对于神经网络来说，输入 input 应该是 number 类型的，因为神经网络里的每个 neuron 都是做一些加减乘除操作。但是真实世界的数据的类型是多种多样的，有数字类型(number)，有分类类型(categorical)，分类类型通常会用 vector 来替代。

feature columns 的作用是：**feature columns bridge raw data with the data your model needs**。

![图片](https://3.bp.blogspot.com/-3Wf_6BEn7GE/Wg4GiQ9TXDI/AAAAAAAAEGo/yoLiIyJW1c4Vh-VfP4vVjuaD92rcnVphACLcBGAs/s1600/2.jpg)


feature columns package 的一些方法：
![图片](https://4.bp.blogspot.com/-geC5Hmnhtto/Wg4Gsl15_MI/AAAAAAAAEGs/A8idgWJnjUASspX_JRiOOqiykVl_LD7VwCLcBGAs/s1600/3.jpg)

从图中可以看出，Categorical Column 和 Dense Column 各自有一些方法，bucketized_column 继承了这两个类:

- *Numeric Column*: 将原数据转换成各种数据类型（默认是 tf.float32，可以转换成 tf.float 64、矩阵等）
- *Bucketized Column*: 将原数据归类到各个 Bucket 中
![图片](https://2.bp.blogspot.com/-qrTI2ZUBr7w/Wg4G9lWHk5I/AAAAAAAAEG0/v17Zqcix1Wou5ZRpTGxAQ8jMSBjCKmCAACLcBGAs/s1600/4.jpg)


  Date Range Represented as:
  |  年份   | 经过 feature column 转换  |
  |  ----  | ----  |
  | 1960  | [1, 0, 0, 0] |
  | >= 1960 but < 1980  | [0, 1, 0, 0] |
  | >= 1980 but < 2000  | [0, 0, 1, 0] |
  | > 2000 | [0, 0, 0, 1] |
- *Categorical Vocabulary Column*: 用 one-hot vector 表示字符串

  ![图片](https://1.bp.blogspot.com/-tATYn91S0Mw/Wg4HVJgTy6I/AAAAAAAAEG8/I0GiWJH0aBYSwfuyBFGwRiS0SHVVGrNngCLcBGAs/s1600/6.jpg)

- *Indicator columns*: 每一类别用 one-hot vector 中的一维来表示
- *Embedding columns*: 相比起 Indicator columns 用一维来表示一个类别（会造成整个 vector 的维数太多），Embedding columns 可以用更少的维数来表示一个类别

![图片](https://2.bp.blogspot.com/-q7GLL9Z95uY/Wg4KIyRryYI/AAAAAAAAEHc/BckVSXOmT1M0qs79D60t2XMv1RFNSd89gCLcBGAs/s1600/image9.jpg)

推荐阅读：
- [Introducing tensorflow feature columns](https://developers.googleblog.com/2017/11/introducing-tensorflow-feature-columns.html)
- [Demonstration of tensorflow feature columns](https://medium.com/ml-book/demonstration-of-tensorflow-feature-columns-tf-feature-column-3bfcca4ca5c4#id_token=eyJhbGciOiJSUzI1NiIsImtpZCI6IjA3NGI5MjhlZGY2NWE2ZjQ3MGM3MWIwYTI0N2JkMGY3YTRjOWNjYmMiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20iLCJuYmYiOjE2NTg5NzcwNjksImF1ZCI6IjIxNjI5NjAzNTgzNC1rMWs2cWUwNjBzMnRwMmEyamFtNGxqZGNtczAwc3R0Zy5hcHBzLmdvb2dsZXVzZXJjb250ZW50LmNvbSIsInN1YiI6IjEwNzk4MjcxNjIxMTc0ODc2NDE0MyIsImVtYWlsIjoiZGVsYXByYWRhemhhb0BnbWFpbC5jb20iLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiYXpwIjoiMjE2Mjk2MDM1ODM0LWsxazZxZTA2MHMydHAyYTJqYW00bGpkY21zMDBzdHRnLmFwcHMuZ29vZ2xldXNlcmNvbnRlbnQuY29tIiwibmFtZSI6ImFsaWNlIHpoYW8iLCJwaWN0dXJlIjoiaHR0cHM6Ly9saDMuZ29vZ2xldXNlcmNvbnRlbnQuY29tL2EvQUl0YnZtbGwzUFM3cm52OXpTYXpiaXl1eG51NXRJanNqN3NCT0Z6Z3Z0R1E9czk2LWMiLCJnaXZlbl9uYW1lIjoiYWxpY2UiLCJmYW1pbHlfbmFtZSI6InpoYW8iLCJpYXQiOjE2NTg5NzczNjksImV4cCI6MTY1ODk4MDk2OSwianRpIjoiMWQyMmIxMTk2YjkwOTlhMzE2MDM2ZGJjOWNhNTZhNmNkNzcxZmQ1NyJ9.Sc6QCUsOt4pSzOPYlRnRtdxeK0sO0vRt6B0IgJ4xXgpLgh9yxD5Blr0G774xZ2i0Sht-n8Cxo5tSAs3nvu7SIA0Sev9vt9r7g75Hpv-rQX6Q2EzviNmSE08iP43POx74hTzkoEWvq7sDfnv7vgvqDJ2yY3LTYfmb_ZC43_uzidLJIZjM7RUbeEhoydfan4g61U4e8goyaOae75oZZ3SjcN5bYxEEo0ITNxT_fuumyjYt81gR3S1cvsNTMZVVEXKT6YpOqEJmd8Ci6hhWk41JPwdtbmmAcLBEiUy5b5mvUUghTBD5y9t00Agb-Ile_b6MTylVuQe7yT56Yxhds6VrmA)
### 创建模型

```python
def create_model(my_learning_rate, feature_layer, my_metrics):
    model = tf.keras.models.Sequential();
    
    model.add(feature_layer);
    
    # units 和 input_shape 为什么是 1 ？
    model.add(tf.keras.layers.Dense(units=1, input_shape=(1,), activation=tf.sigmoid),);
    
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr = my_learning_rate),
                  loss = tf.keras.losses.BinaryCrossentropy(),
                  metrics = my_metrics);
    
    return model;
```

创建模型时将刚刚通过 feature columns 创建的 feature layer 添加到 model 中，然后再添加一层 neuron，激活函数选择 sigmoid。

并为模型选择 gradient decent 方法（此处选择的是 RMSprop 方法），因为题目是分类问题，所以损失函数选择的是 CrossEntropy。


### 训练模型

```python
def train_model(model, dataset, epochs, label_name, batch_size = None, shuffle = True):
    features = {name: np.array(value) for name, value in dataset.items()};
    label = np.array(features.pop(label_name));
    history = model.fit(x=features, y=label, batch_size=batch_size, epochs=epochs, shuffle=shuffle);
    
    epochs = history.epoch;
    
    hist = pd.DataFrame(history.history);
    
    return epochs, hist;
```

将数据传入模型进行训练。

### 画出 accuracy 随着 epoch 的进行的变化

```python
# plot the relation between epoch and one or more classification metrics
def plot_curve(epochs, hist, list_of_metrics):
    plt.figure();
    plt.xlabel("Epoch");
    plt.ylabel("Value");
    
    for m in list_of_metrics:
        x = hist[m];
        plt.plot(epochs[1:], x[1:], label=m);
    
    plt.legend();
```

### Main 函数

```python
# main function
learning_rate = 0.001;
epochs = 20;
batch_size = 100;
label_name = "median_house_value_is_high";
classification_threshold = 0.52; # metrics 的 cut point

# Establish the metrics the model will measure.
METRICS = [
#            tf.keras.metrics.BinaryAccuracy(name='accuracy', 
#                                            threshold=classification_threshold),
#            tf.keras.metrics.Precision(name='precision', thresholds=classification_threshold),
#            tf.keras.metrics.Recall(name='recall', thresholds=classification_threshold),
            tf.keras.metrics.AUC(num_thresholds=100, name='auc'),
          ];

my_model = create_model(learning_rate, feature_layer, METRICS);

epochs, hist = train_model(my_model, train_df_norm, epochs, label_name, batch_size, True);

list_of_metrics_to_plot = ["auc"];

plot_curve(epochs, hist, list_of_metrics_to_plot);
```

只看 accuracy 在数据集分布不均时是不能够正确衡量模型的性能的，所以可以通过看其他指标，如 precision, recall, auc 来衡量模型性能。










