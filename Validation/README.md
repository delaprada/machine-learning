# Validation
[course](https://colab.research.google.com/github/google/eng-edu/blob/main/ml/cc/exercises/validation_and_test_sets.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=validation_tf2-colab&hl=zh-cn#scrollTo=nd_Sw2cygOip)

## knowledge

### shuffle
当数据集本身分布不均衡的时候，可能导致 validation data 的 loss 总是比 training data 的 loss 要大，因此可以做 shuffle，将数据集打乱，这样将数据集分割成 training 和 validation 的时候数据分布就不会不均衡：
```
shuffled_train_df = train_df.reindex(np.random.permutation(train_df.index));
```