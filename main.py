import tensorflow as tf
import tensorflowjs as tfjs
import numpy as np
import pandas as pd
import os
import datetime
from tensorflow import keras
from tensorflow.keras import layers

saved_tf_models_dir = os.path.join(os.getcwd(), 'saved_tf_models')
os.makedirs(saved_tf_models_dir, exist_ok=True)

saved_tfjs_models_dir = os.path.join(os.getcwd(), 'saved_js_models')
os.makedirs(saved_tfjs_models_dir, exist_ok=True)


os.environ['CUDA_VISIBLE_DEVICE'] = '-1'
if tf.test.gpu_device_name():
    print('GPU found')
else:
    print('No GPU found')

# read data
dataset_df = pd.read_csv('car_data.csv')

# clean data
dataset_df = dataset_df.iloc[:, 1:]
dataset_df['current_year'] = datetime.datetime.today().year
dataset_df['car_age'] = dataset_df['current_year'] - dataset_df['Year']
dataset_df.drop(['Year'], axis=1, inplace=True)
dataset_df = dataset_df.dropna()

# delete unwanted columns
dataset_df = dataset_df.drop(['current_year'], axis=1)

# converts categorical data into dummy or indicator variables.
dataset_df = pd.get_dummies(dataset_df, drop_first=True)

# type cast data uint8 to int64
dataset_df['Fuel_Type_Diesel'] = tf.cast(
    dataset_df['Fuel_Type_Diesel'], dtype=tf.int64)
dataset_df['Fuel_Type_Petrol'] = tf.cast(
    dataset_df['Fuel_Type_Petrol'], dtype=tf.int64)
dataset_df['Seller_Type_Individual'] = tf.cast(
    dataset_df['Seller_Type_Individual'], dtype=tf.int64)
dataset_df['Transmission_Manual'] = tf.cast(
    dataset_df['Transmission_Manual'], dtype=tf.int64)

# split data
train_dataset = dataset_df.sample(frac=0.8, random_state=0)
test_dataset = dataset_df.drop(train_dataset.index)

# Split features from labels
train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('Selling_Price')
test_labels = test_features.pop('Selling_Price')

# Normalization
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))
print(normalizer.mean.numpy())

first = np.array(train_features[:1])

test_results = {}
# Build and Compile model


def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(loss='mean_absolute_error',
                  optimizer=tf.keras.optimizers.Adam(0.001))
    return model


dnn_model = build_and_compile_model(normalizer)
print("================================ DNN model architecture ================================\n")
dnn_model.summary()

test_results['dnn_model'] = dnn_model.evaluate(
    test_features, test_labels, verbose=0)

pd.DataFrame(test_results, index=['Mean absolute error [Selling_Price]']).T

prediction = dnn_model.predict([[9.85, 6900, 0, 6, 0, 1, 0, 1]])

print(f'\nPredicted result using Tensorflow Model: {prediction}\n')

# save the model as a SavedModel format
tf.saved_model.save(dnn_model, './saved_tf_models/my_model')

# convert TF model saveModel to TFJS model
tfjs.converters.convert_tf_saved_model(
    './saved_tf_models/my_model', './saved_js_models/my_model_1')

# Convert the TF model from the "model" variable above to a TFJS model
tfjs.converters.save_keras_model(dnn_model, './saved_js_models/my_model_2')
