# Neural Network

此任务主要是简单地搭建一个神经网络模型并进行训练，涉及如下知识点：
- normalization
- feature columns
- feature cross
- dropout

[google crash course link](https://developers.google.com/machine-learning/crash-course/introduction-to-neural-networks/programming-exercise?hl=zh-cn)

## 环境

- python: 3.7.13
- pandas: 1.3.5
- numpy: 1.21.5
- tensorflow: 2.0.0
- matplotlib: 3.5.1
- seaborn: 0.11.2

## 数据集

训练过程使用的数据集是 google 提供的 [California Housing Dataset](https://developers.google.com/machine-learning/crash-course/california-housing-data-description)

## 知识点

### feature cross

- 含义：拿到一堆 feature 往 linear model 里面 train 时候，可以单拎每一个 feature 成为一个维度，也可以把这些 feature 组合一下成为一个维度。组合的方式就是将 feature 乘起来。

- 为什么要做 feature cross：当现有的 feature set 线性不可分，而做了 feature cross 之后则线性可分了。

- 什么 feature 适合做 cross：**correlated**，比如一个 label 需要一组相关联的特征共同决定，则可以尝试将这组特征 cross 一下（比如 latitude 和 longitude）。

- 所有 model 都需要这个步骤么：不是所有 model 都需要，一些 nonlinear model（比如 deep learning nn），这个步骤就没有必要，因为 model 自身可以学习 feature 之间的 nonlinear relationship。

例如在这个 practice 中，就将 latitude 和 longitude 属性做了 feature cross：
```python
# 先将 latitude 和 longitude 各自 bucketize，再做 feature cross

latitude_as_a_numeric_column = tf.feature_column.numeric_column('latitude'); # 创建新的 feature_column
latitude_boundaries = list(np.arange(
    int(min(train_df_norm['latitude'])),
    int(max(train_df_norm['latitude'])),
    resolution_in_Zs)); # 根据 latitude 取值范围的最大值、最小值、bucket 间隔划分 latitude 的 boundaries
latitude = tf.feature_column.bucketized_column(latitude_as_a_numeric_column, latitude_boundaries); # bucketize 之后的 feature_column

longitude_as_a_numeric_column = tf.feature_column.numeric_column('longitude'); # 创建新的 feature_column
longitude_boundaries = list(np.arange(
    int(min(train_df_norm['longitude'])),
    int(max(train_df_norm['longitude'])),
    resolution_in_Zs)); # 根据 latitude 取值范围的最大值、最小值、bucket 间隔划分 latitude 的 boundaries
longitude = tf.feature_column.bucketized_column(longitude_as_a_numeric_column, longitude_boundaries); # bucketize 之后的 feature_column

# 创建基于 latitude 和 longitude 的 feature cross
latitude_x_longitude = tf.feature_column.crossed_column([latitude, longitude], hash_bucket_size = 100);
crossed_feature = tf.feature_column.indicator_column(latitude_x_longitude);
```

### dropout

参考链接：[深度学习中 dropout 原理解析](https://zhuanlan.zhihu.com/p/38200980)

#### dropout 出现的原因

在机器学习的模型中，如果模型的参数太多，而训练样本又少，就会导致训练出来的模型容易产生**过拟合**的现象。

为了解决过拟合的问题，一般采用模型集成的方法，即训练多个模型进行组合，这又会导致**训练费时**问题。


#### 什么是 dropout

dropout 作为训练神经网络时的一种 trick 选择，在每个训练批次中，通过忽略一半的 neuron 节点，可以明显地减少过拟合现象。这种方式可以减少 neuron 之间的相互作用。

简而言之，在神经网络做前向传播时，让某些神经元以概率 p 停止工作，这样可以使模型泛化性能更强，因为他不会依赖某些局部特征。

![](https://pic2.zhimg.com/80/v2-5530bdc5d49f9e261975521f8afd35e9_1440w.jpg)

#### dropout 的具体工作流程

标准的神经网络如下：

![](https://pic3.zhimg.com/80/v2-a7b5591feb14da95d29103913b61265a_1440w.jpg)

正常的流程是：我们首先把x通过网络前向传播，然后把误差反向传播以决定如何更新参数让网络进行学习。

dropout 之后的神经网络如下：

![](https://pic3.zhimg.com/80/v2-24f1ffc4ef118948501eb713685c068a_1440w.jpg)

新的流程变成：
- 首先随机（临时）删掉网络中一半的隐藏神经元，输入输出神经元保持不变（图3中虚线为部分临时被删除的神经元）
- 然后把输入x通过修改后的网络前向传播，然后把得到的损失结果通过修改的网络反向传播。一小批训练样本执行完这个过程后，在没有被删除的神经元上按照随机梯度下降法更新对应的参数（w，b）

然后继续重复这一过程：
- 恢复被删掉的神经元（此时被删除的神经元保持原样，而没有被删除的神经元已经有所更新）
- 从隐藏层神经元中随机选择一个一半大小的子集临时删除掉（备份被删除神经元的参数）
- 对一小批训练样本，先前向传播然后反向传播损失并根据随机梯度下降法更新参数（w，b） （没有被删除的那一部分参数得到更新，删除的神经元参数保持被删除前的结果）。

#### dropout 在神经网络中的使用

怎么实现让一些神经元以一定的概率停止工作：

1. 在训练模型阶段，给每个神经元都添加一道概率流程：
![](https://pic3.zhimg.com/80/v2-543a000fcfe9778cd64c898c01743aae_1440w.jpg)

    对应的公式变化：
    - 没有 dropout 的网络计算公式：
    ![](https://pic1.zhimg.com/80/v2-11fd2a086d59490cf8121d90c4ec4e68_1440w.jpg)
    - 有 dropout 的网络计算公式：
    ![](https://pic4.zhimg.com/80/v2-61933b0548270880aa5d5497ede5b383_1440w.jpg)

公式中的 Bernoulli 函数随机生成 0, 1 向量（以概率 p 生成 0）。

2. 在测试模型阶段：

    测试模型时，每个神经元的输出单元的权重都要乘 p。

    ![](https://pic4.zhimg.com/80/v2-335782876686a248b51ff739c7e9b1ff_1440w.jpg)

    这么做才能让神经元在测试时的期望和训练时的期望一致。

#### 为什么说 dropout 可以解决过拟合

1. 取平均的作用：整个dropout过程就相当于对很多个不同的神经网络取平均。而不同的网络产生不同的过拟合，一些互为“反向”的拟合相互抵消就可以达到整体上减少过拟合。

2. 减少神经元之间复杂的共适应关系：因为dropout程序导致两个神经元不一定每次都在一个dropout网络中出现。这样权值的更新不再依赖于有固定关系的隐含节点的共同作用，阻止了某些特征仅仅在其它特定特征下才有效果的情况 。迫使网络去学习更加鲁棒的特征。

3. Dropout类似于性别在生物进化中的角色：物种为了生存往往会倾向于适应这种环境，环境突变则会导致物种难以做出及时反应，性别的出现可以繁衍出适应新环境的变种，有效的阻止过拟合，即避免环境改变时物种可能面临的灭绝。




