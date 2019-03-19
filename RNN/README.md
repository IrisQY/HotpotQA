# HotpotQA: RNN Model
It takes about 1 hour to preprocess the data on NV6 GPU, and 20-30 hours to train the model.
### Preprocess the data
```Python
python main.py --mode prepro --data_file hotpot_train_v1.1.json --para_limit 2250 --data_split train --num_per_bucket 2000
python main.py --mode prepro --data_file hotpot_dev_distractor_v1.json --para_limit 2250 --data_split dev --num_per_bucket 100000000
```
