---
layout: post
title: Forecasting Hourly MRT Traffic using MLP
date: 2018-01-25 13:32:20 +1200
description: post for my forecasting attempt at predicting train traffic using MLP
img:  project-forecasting-mlp/forecast_thumbnail.jpg
tags: [Neural Networks, Time-Series Forecasting, MLP, LSTM]
---

Accenture also hosted a forecasting challenge in which I had no interest in, until I saw it had a cash prize of 60,000 pesos… WTF, I am in.  The goal of this forcast is to build a MLP model that can predict an hourly entries or exit on any given MRT station.

### Steps

These steps serves as my guide to planning out the project, this is by no means the best approach when it comes to creating forecasting models but it is relatively simple which is good for retards like me.

 - step 1: Gather Data and Determine Features to use as Inputs
 - Step 2: Build a base model to compare metrics and results (Build a simple persistance model)
 - Step 3: Neural Network Model building. Use a large grid search to find good hyperparameters (Do this if you have no idea how to tune nerual networks)
 - Step 4: Finetune Neural Networks and compare results with base model(Persistance model)
 - Step 5: Try its Performance on the test set.

### Step 1: Gathering Data and Determining Feature Inputs

I havent had much experience dealing with forecasting models, so I am a bit aware of my sad limitations. Working on a tight schedule I had no time to study how to diagnose time-series data and which forecasting model to use with it. I have absolutely no idea how popular forecasting methods such as arima, sarmia or VEC work (yikes).

This leave me to rely upon Neural Networks as my approach, the reasoning for this is that neural net can optimize highly non-linear functions and can also receive multiple inputs (Multivariate). For this forecast, I use a multilayer perceptron to biuld my neural network. I used the same data as the one I gathered to build my [visualization entry](https://ryanliwag.github.io/Visualizing-MRT-2017/). The hourly data I have is sadly in excel format, making it troublesome to extract data, that's why I will only be working with 3 months of data. I do split the data into 3 pieces for my training, validation and testing (85:10:5)

 Features to use (Inputs to the model)
 - Week of the month
 - Day of the Week
 - Holiday
 - Amount of People previous Hour

### Step 2: Building a Persistance Model

The metric commonly used for forcasting is Mean squared error(MSE), so I want  to establish a base line to which to compare my MLP model. Persistance models are very simple forcast where previous values is shifted by 1, the previous value an hour ago persist to the next hour.

```python
# Basic Persitance Model
dataset = read_csv("dataset-1.csv")
dataset_ = dataset[["time", "entries"]]

scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(dataset_)
dataset_scaled = scaler.transform(dataset_)
data_naive = pd.DataFrame(dataset_scaled)

data_frame = concat([data_naive.shift(1), data_naive], axis=1)
data_frame.columns = ['t', 't+1', "entries", "entries+1"]
dataframe = data_frame[["t+1","entries+1"]]

# split into train and test sets
X = dataframe.values
train_size = 1520 # Dataset Cut
train, test = X[1:train_size], X[train_size:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]

# Naive Model
def model_persistence(x):
	return x

# walk-forward validation
predictions = list()
for x in test_X:
	yhat = model_persistence(x)
	predictions.append(yhat)
test_score = mean_squared_error(test_y, predictions)
print('Test MSE: %.3f' % test_score)
```

I will be using the same scaled dataset when building the MLP model where each entry is ranged from (-1,1), this will make it so that the MSE of the persistance model is relative to the same scale.

The model optained an mse of **0.109** on the validation set.

![naive_model]({{site.baseurl}}/assets/img/project-forecasting-mlp/naive_model.png)


### Step 3: Building the MLP Model

So when it comes to building an MLP model there is no clear cut guide, to setting the hyperparameters. For this task, if you have a bit of time to kill like me. You can opt to using a massive grid search for hyperparameters. Keras Deep learning library has a compatible grid search extension in scikit-learn. This process does take a while, this was done on an I5 intel cpu which lasted approximetly a day. I havent managed to try gpu processing cause installing cuda in linux is a goddamn nightmare.

GRID Search settings

| Parameter | Values |
| ------ | ------ |
| Epochs | [500, 800, 1000] |
| Batch Size | [16, 32, 64] |
| Neurons | [1, 2, 4, 6] |
| Learning Rate (LR) | [0.001, 0.005, 0.01, 0.02] |
| LR Decay  | [0.01, 0.01, 0.02] |
| Dropout | [0.1, 0.2, 0.4] |

Python Code
```python

def model_mlp(neurons=1, dropout_rate=0.0):
    model = Sequential()
    model.add(Dense(neurons, activation='relu', input_dim=train_X.shape[1]))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9,
                                 beta_2=0.999, epsilon=None,
                                 decay=0.2, amsgrad=False)
    model.compile(loss='mean_squared_error', optimizer=adam)
    return model

model = KerasClassifier(build_fn=model_mlp, verbose=0)

batch_size =  [16, 32, 64]
epochs =  [500, 800, 1000]
neurons = [1, 2, 4, 6]
dropout_rate = [0.1, 0.2, 0.4]

param_grid = dict(batch_size=batch_size, epochs=epochs, neurons=neurons,
                 dropout_rate=dropout_rate)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-2)
grid_result = grid.fit(train_X, train_y)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
```

### Step 4: Finetune Neural Networks and compare results with base model

After finding some suitable hyperparameters, I still tried to finetune the model so that it may perform better on the validation data set.

The MLP model obtained a mse score of **0.0178** on the validation set

![mlp_model]({{site.baseurl}}/assets/img/project-forecasting-mlp/mlp_model.png)

It looks like its outperforming the persistance model. But this is the case for single step forecast, where the model simply tries to forecast 1 step or in my case 1 hour after. I was curious to how the model would react to multistep forecast, where previous forecast are being used as inputs. Ofcourse the error will be accumulated, but I am just curious to see the results. I made a small function to try forecasting multiple step from just 1 previous value, inputs such as dates and holidays are already known, and predicted inputs are used to predict the next value.

```python

# inputs refer to other features such as dates, week, Holidays
# X will be the first forecast in which the model will start off with
def predict_multi_step(model, X, inputs):
    predicts = list()
    predicts.append(X)

    for i in range(len(inputs)):
        test = np.append(inputs[i], predicts[i])
        forecast_val = model.predict(np.expand_dims(test, axis=0))
        predicts.append(forecast_val)

    return predicts[1:]

```

The result is kind of expected since the value is repeated, the error is accumulated after each succeding prediction and the pattern is repeated also slightly decreasing over time. Verdict: This method may work if you are trying to forcast a smaller amount of steps, such as maybe all the hours for the day.

![mlp_model_multistep]({{site.baseurl}}/assets/img/project-forecasting-mlp/mlp_multi-step.png)

### Step 5: Model performance on testing set

So my last test for the model using my test set which is hourly data of 5:00am to 10:00pm on april 1 till april 12.

Again with the persistance model as my baseline mse it scored **0.078**, while the MLP model scored **0.0148**.

![mlp_model_test_result]({{site.baseurl}}/assets/img/project-forecasting-mlp/mlp_model_final_plot.png)

### Further Improvements

If I wasnt limited with the extraction of data (I only have 3 months and a half) due to it being in weird excel format, and I would have liked to see the result if I include yearly data. Cause I tried plotting the total traffic and they do change over the months. Also trying a different neural network such as LSTM might be helpfull, possibly because prediction in forcast relies on historical context.

Download the whole dataset from here: https://drive.google.com/drive/folders/1D0eYzBV4jewEIosX7F-iOIFUMgk6TvrD?usp=sharing

Data Cleaning process on this notebook: https://github.com/ryanliwag/ryanliwag.github.io/blob/master/notebooks/Data%20Cleaning.ipynb

Model building process on this notebook: https://github.com/ryanliwag/ryanliwag.github.io/blob/master/notebooks/MLP_model.ipynb
