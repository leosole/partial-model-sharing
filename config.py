client1 = {
    'train_split': 110000, 'initial_split': 0, 'initial_test': 220000, 'test_split': 284807, 'batch_size': 128, 'label': 30, 'columns': [*range(1,22)]
}
client2 = {
    'train_split': 220000, 'initial_split': 110000, 'initial_test': 220000, 'test_split': 284807, 'batch_size': 128, 'label': 30, 'columns': [*range(12,30)]
}
server = {
    'train_split': 220000, 'initial_split': 0, 'test_split': 284807, 'batch_size': 128, 'label': 30
}
shared_layers = [64, 64, 8]
ind_layers = [32, 32, 8]
agg_layers = [64, 64, 1]
local_layers = [96, 96, 64, 64, 1]
server_layers = [29, 96, 96, 64, 64, 1]
rounds = 50
epochs = 1
random = 13