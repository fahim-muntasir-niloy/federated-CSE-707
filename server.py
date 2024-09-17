import flwr as fl
import numpy as np

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd: int,
        results,
        failures,
    ):
        weights = super().aggregate_fit(rnd, results, failures)
        if weights is not None:
            # Save weights
            print(f"Saving round {rnd} weights...")
            np.savez(f"round-{rnd}-weights.npz", *weights)
        return weights


strategy = SaveModelStrategy()



fl.server.start_server(server_address="0.0.0.0:8080", 
                       config=fl.server.ServerConfig(num_rounds=50),
                       strategy=strategy
                       )