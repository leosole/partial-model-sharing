#%%
import flwr as fl
import sys
sys.path.append(".")
import config

strategy = fl.server.strategy.FedAvg(min_available_clients=2,min_fit_clients=2,fraction_fit=1.0,fraction_eval=1.0)
fl.server.start_server(config={"num_rounds": config.rounds}, strategy=strategy)
# %%\
