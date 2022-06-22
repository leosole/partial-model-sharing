batch_size = 2048
# client1 = {
#     'train_split': 100000, 'initial_split': 0, 'initial_test': 200000, 'test_split': 284807, 'batch_size': batch_size, 'label': 30, 'columns': [0, 1, 4, 6, 7, 8, 12, 13, 14, 15, 16, 17, 19, 22, 23, 24, 25, 26, 27, 28, 29],
# }
# client2 = {
#     'train_split': 200000, 'initial_split': 100000, 'initial_test': 200000, 'test_split': 284807, 'batch_size': batch_size, 'label': 30, 'columns': [0, 2, 3, 5, 8, 9, 10, 11, 13, 15, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
# }
all_col = range(201)
client1 = {
    'train_split': 80000, 'initial_split': 0, 'initial_test': 160000, 'test_split': 200000, 'batch_size': batch_size, 'columns': [x for x in all_col if x % 2 == 0 or x < 100],
}
client2 = {
    'train_split': 160000, 'initial_split': 80000, 'initial_test': 160000, 'test_split': 200000, 'batch_size': batch_size, 'columns': [x for x in all_col if x % 2 != 0 or x < 100]
}
server = {
    'train_split': 160000, 'initial_split': 0, 'test_split': 200000, 'batch_size': batch_size
}
path = 'data/train_santander_processed.csv'
shared_layers = [512, 256]
ind_layers = [512, 256]
agg_layers = [512, 512, 1]
local_layers = [512, 512, 512, 1]
server_layers = [512, 512, 512, 1]
dropout = 0.1
weight_decay = 0
rounds = 40
epochs = 1
random = 3
freeze = False
resample = False