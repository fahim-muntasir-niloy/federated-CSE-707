import tensorflow as tf
import flwr as fl
import pandas as pd
import numpy as np
from model import tf_model
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error



# dataset
from sklearn.model_selection import train_test_split
df = pd.read_csv("used_mobile_le.csv")

X = df.drop(columns=["normalized_used_price"])
y = df["normalized_used_price"]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.2, random_state=42)


# model
model = tf_model


weights_dict = np.load('round-50-weights.npz')


for layer in model.layers:
    layer_name = layer.name
    if layer_name in weights_dict:
        layer_weights = weights_dict[layer_name]
        model.get_layer(name=layer_name).set_weights(layer_weights)



# evaluation
y_pred = model.predict(X)


mae = mean_absolute_error(y, y_pred)
mape = mean_absolute_percentage_error(y, y_pred)
rmse = root_mean_squared_error(y, y_pred)

print(f"MAE: {mae:.2f}, MAPE: {mape:.2f}, RMSE: {rmse:.2f}")