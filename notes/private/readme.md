[This directory is ignored to let us write private notes for ourselves]: #

# Machine Learning
Some code for debugging from `prepare_data` method
```
figure = sns.pairplot(parsed_dataset[column_names], diag_kind="kde")
figure.savefig("output2.png")

data = pd.read_csv(os.path.dirname(
    os.path.abspath(__file__)) + "/data.csv")
for column, _len in data.count().items():
    if _len < 9000:
        data.drop(column, axis=1, inplace=True)

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
column_names = ["MPG", "Cylinders", "Displacement", "Horsepower", "Weight",
                "Acceleration", "Model Year", "Origin"]
raw_dataset = pd.read_csv(url, names=column_names,
                          na_values="?", comment="\t",
                          sep=" ", skipinitialspace=True)
dataset = raw_dataset.copy()

dataset = dataset.dropna()
dataset["Origin"] = dataset["Origin"].map(
    {1: "USA", 2: "Europe", 3: "Japan"})
dataset = pd.get_dummies(
    dataset, columns=["Origin"], prefix="", prefix_sep="")

# Drop all rows with column value -1
data = data[(data.iloc[:, 1:] != -1).all(axis=1)]
# Drop uknown values
data = data.dropna()
```

# TF Privacy

## Tutorials
 - https://www.tensorflow.org/tutorials/keras/regression
 - https://www.youtube.com/watch?v=-vHQub0NXI4
 - https://blogs.rstudio.com/ai/posts/2019-12-20-differential-privacy/
 - https://github.com/tensorflow/privacy/blob/master/tutorials/Classification_Privacy.ipynb

## Code stuff
```
# Print TF version
print("Tensorflow version:", tf.__version__)
```

```
# Disables TensorFlow 2.x behaviors
tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()
```

# Opacus

## Tutorials
 - https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/

## [FAQ](https://opacus.ai/docs/faq)

## Notes 
 - You should, however, keep in mind that increasing batch size has its price in terms of epsilon, which grows at O(sqrt(batch_size)) as we train (therefore larger batches make it grow faster). The good strategy here is to experiment with multiple combinations of batch_size and noise_multiplier to find the one that provides best possible quality at acceptable privacy guarantee [LINK](https://opacus.ai/tutorials/building_text_classifier)
