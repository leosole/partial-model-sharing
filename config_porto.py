batch_size = 2048
# client1 = {
#     'train_split': 100000, 'initial_split': 0, 'initial_test': 200000, 'test_split': 284807, 'batch_size': batch_size, 'label': 30, 'columns': [0, 1, 4, 6, 7, 8, 12, 13, 14, 15, 16, 17, 19, 22, 23, 24, 25, 26, 27, 28, 29],
# }
# client2 = {
#     'train_split': 200000, 'initial_split': 100000, 'initial_test': 200000, 'test_split': 284807, 'batch_size': batch_size, 'label': 30, 'columns': [0, 2, 3, 5, 8, 9, 10, 11, 13, 15, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
# }
all_col = range(68)
client1 = {
    'train_split': 250000, 'initial_split': 0, 'initial_test': 500000, 'test_split': 595212, 'batch_size': batch_size, 'columns': [x for x in all_col if x % 2 == 0 or x > 40],
}
client2 = {
    'train_split': 500000, 'initial_split': 250000, 'initial_test': 500000, 'test_split': 595212, 'batch_size': batch_size, 'columns': [x for x in all_col if x % 2 != 0 or x > 40]
}
server = {
    'train_split': 500000, 'initial_split': 0, 'test_split': 595212, 'batch_size': batch_size
}
path = 'data/train_portoseguro_processed.csv'
shared_layers = [256, 128]
ind_layers = [256, 128]
agg_layers = [256, 256, 256, 1]
local_layers = [256, 256, 256, 1]
server_layers = [512, 512, 1]
dropout = 0.1
weight_decay = 0
rounds = 100
epochs = 1
random = 11
freeze = False
resample = True