# HotpotQA: RNN Model
It takes about 1 hour to preprocess the data on NV6 GPU, and 20-30 hours to train the model.
### Preprocess the data
```sh
python main.py --mode prepro --data_file hotpot_train_v1.1.json --para_limit 2250 --data_split train --num_per_bucket 2000
python main.py --mode prepro --data_file hotpot_dev_distractor_v1.json --para_limit 2250 --data_split dev --num_per_bucket 100000000
python main.py --mode prepro --data_file hotpot_dev_fullwiki_v1.json --data_split dev --fullwiki --para_limit 2250 --num_per_bucket 100000000

```
### Train the model
```sh
CUDA_VISIBLE_DEVICES=0 python main.py --mode train --para_limit 2250 --batch_size 12 --init_lr 0.001 --keep_prob 0.8 --sp_lambda 1.0 --patience 2 --total_num_of_buckets 9 --checkpoint 1000
```
Note that "total_num_of_buckets" should match the number of ".pkl" files generated by preprocess.
### Predict and evaluate
```sh
python main.py --mode test --data_split dev --para_limit 2250 --batch_size 12 --init_lr 0.001 --keep_prob 1.0 --sp_lambda 1.0 --patience 2 <name of folder> --prediction_file dev_distractor_pred.json
python hotpot_evaluate_v1.py dev_distractor_pred.json hotpot_dev_distractor_v1.json
```
