# flower
data: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud - should be saved in data/creditcard.csv

## Transfer Learning
```
python server.py
python transfer_client.py 1
python transfer_client.py 2
```
## Partial Model Sharing
```
python server.py
python client.py 1
python client.py 2
```
## Individual Training
```
python local_client.py 1
python local_client.py 2
```
## Global Training
```
python global_client.py
```
