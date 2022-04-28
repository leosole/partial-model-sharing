#%%
import flwr as fl

strategy = fl.server.strategy.FedAvg(min_available_clients=2,min_fit_clients=2,fraction_fit=1.0,fraction_eval=1.0)
fl.server.start_server(config={"num_rounds": 50}, strategy=strategy)
# %%\
