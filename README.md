# flower
## config files
To run simulations you must create a config.py file like [this example](config_ieee_tf.py), and import it
```
from [you_config_file] import *
```
In order to run the federated learning experiments, you can run 3 separate terminal tabs, one for [server.py](server.py) and the other two for the training script. 
You can also run the two training scripts in a single terminal tab like below
```
python training_script.py 1 &
python training_script.py 2
```
The training scripts will take a client number as argument (1 or 2)
## TensorFlow examples
The TensorFlow examples are working better than the Pytorch ones
### Transfer Learning
```
python server.py
python tf_transfer.py 1
python tf_transfer.py 2
```
### Partial Model Sharing
```
python server.py
python tf_partial.py 1
python tf_partial.py 2
```
### Individual Training
```
python tf_local.py 1
python tf_local.py 2
```
### Global Training
```
python tf_global.py
```
## Pytorch examples
### Transfer Learning
```
python server.py
python transfer_client.py 1
python transfer_client.py 2
```
### Partial Model Sharing
```
python server.py
python client.py 1
python client.py 2
```
### Individual Training
```
python local_client.py 1
python local_client.py 2
```
### Global Training
```
python global_client.py
```
