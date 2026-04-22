### Download dataset from Huggingface
bash data/download.sh inter

### Venv
- cd to repo
- uv init
- uv venv
- source .venv/bin/activate
- uv pip install -r requirements.txt

### Test run
python3 train_demo.py --mode inter --lr 1e-4 --batch_size 8 --trainN 5 --N 5 --K 1 --Q 1 --train_iter 100 --val_iter 50 --test_iter 100 --val_step 100 --max_length 64 --model proto

### Model args
```shell
-- mode                 training mode, must be inter, intra, or supervised
-- trainN               Num of entities types during training
-- N                    Num of entities types during val and test (higher is harder)
-- K                    Num of support examples of each entity type in train, val and test (higher is easier)
-- Q                    Num of query per entity type i.e. how many samples the model get graded on (higher means more gradient signal per episode, smoother training)
-- batch_size           I think it's how many episodes is in a batch
-- train_iter           num of batches during training (we take a step after each batch)
-- val_step             how many train batches between each pause to validate on val step. 
-- val_iter             num of validation batches to evaluate when we pause. We weights are the point of the best val score is the final model.
-- test_iter            num of test episodes batches to evaluate model and get final results.
-- model                model name, must be proto
-- max_length           max length of tokenized sentence
-- lr                   learning rate
-- weight_decay         weight decay
-- grad_iter            accumulate gradient every x iterations
-- load_ckpt            path to load model (use at test-time to use an already finetuned model)
-- save_ckpt            path to save model (fallback automatically creates a name from N, K and mode)
-- only_test            no training process, only test
-- ckpt_name            checkpoint name
-- seed                 random seed
```