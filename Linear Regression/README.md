# Linear Regression

[course link](https://developers.google.com/machine-learning/crash-course/first-steps-with-tensorflow/programming-exercises?hl=zh-cn)
[keras api link](https://keras.io/api/models/model_training_apis/#compile-method)

## Code Structure

整个代码由 4 个函数构成：
1. `build_model`：使用 `tf.keras.Sequential` api 构建线性模型
2. `train_model`: 使用构建好的 model, 定义好的 epoch, batch_size 训练模型，返回训练出来的 w, b 以及训练的 epoch 和 root_mean_square_error
3. `plot_the_model`：使用 matplotlib 库的方法用散点图画出 train_data 中 x 和 y 的关系，画出训练出来的线性模型
4. `plot_the_loss_curve`：画出每个 epoch 对应的 rmse，可以看出整个 loss 趋势是否收敛（converge）

## Knowledge

### What is RMSprop



### Loss vs. Metrics

- `Loss`: The loss function is used to optimize your model. This function's result will get minimized by the optimizer during the training process, such as: mean square error(linear model), cross-entropy(classification)
- `Metrics`: A metric is used to judge the performance of your model. This is only for you to look at and has nothing to do with the optimization process, such as: root mean square error(linear model), AUC(classification)

比如在 datafountain 比赛里，会使用某个公式评价你的 model 的 performance，这个就是 metrics，这个 metrics 是在你整个模型已经训练完毕之后才会用到的。训练过程使用 metrics 的话，一般是将整个 training dataset 分割成训练集和测试集，使用测试集评价 model。

