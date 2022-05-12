batch_size = 2048
client1 = {
    'train_split': 100000, 'initial_split': 0, 'initial_test': 200000, 'test_split': 284807, 'batch_size': batch_size, 'label': 30, 'columns': [0, 1, 4, 6, 7, 8, 12, 13, 14, 15, 16, 17, 19, 22, 23, 24, 25, 26, 27, 28, 29],
}
client2 = {
    'train_split': 200000, 'initial_split': 100000, 'initial_test': 200000, 'test_split': 284807, 'batch_size': batch_size, 'label': 30, 'columns': [0, 2, 3, 5, 8, 9, 10, 11, 13, 14, 15, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
}
server = {
    'train_split': 200000, 'initial_split': 0, 'test_split': 284807, 'batch_size': batch_size, 'label': 30
}
shared_layers = [128, 128, 128]
ind_layers = [128, 128, 128]
agg_layers = [128, 128, 1]
local_layers = [352, 352, 352, 1]
server_layers = [30, 352, 352, 352, 1]
dropout = 0.3
weight_decay=0
rounds = 70
epochs = 1
random = 2
freeze = False