import tensorflow as tf
import flwr as fl
import pandas as pd
from model import tf_model



# dataset
from sklearn.model_selection import train_test_split
df = pd.read_csv("apple_windows.csv")
X = df.drop(columns=["normalized_used_price"])
y = df["normalized_used_price"]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.2, random_state=200)


# model
model = tf_model

model.compile(
    optimizer='adam',
    loss='mape',
    metrics=['mape']
)


# client object
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()
    
    def fit(self, parameters, config):
        model.set_weights(parameters)
        fit_msg = model.fit(X_train, y_train, 
                  epochs=50,
                  batch_size=32,
                  validation_data=(X_test, y_test), verbose=1)

        print("Fit history: ", fit_msg.history)
        
        return model.get_weights(), len(X_train), {}
    
    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, mape = model.evaluate(X_test, y_test, verbose=1)
        print("Eval MAPE: ", mape)
        return loss, len(X_test), {'accuracy': mape}
    
    
    
fl.client.start_numpy_client(server_address="127.0.0.1:8080",
                             client=FlowerClient().to_client())