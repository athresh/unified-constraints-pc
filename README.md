# unified-constraints-pc
 
  
### To Run
Create a config file for your experiment, specifying the model to be used and the hyperparameters, inside `configs/`.

Currently supported model classes:
- RatSpn
- EinsumNet
- EinsumFlow

Use the following command to launch the experiment:

```python
 python train_ucpc.py --config_file=PATH_TO_CONFIG
```
