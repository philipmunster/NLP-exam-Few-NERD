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

### ProtoLlama (Phase 3+)

#### BERT Baseline (reference)
```shell
python3 train_demo.py --mode inter --encoder_family bert \
  --lr 1e-4 --batch_size 8 --trainN 5 --N 5 --K 1 --Q 1 \
  --train_iter 600 --val_iter 100 --test_iter 500 --val_step 20 \
  --max_length 64 --model proto
```

#### ProtoLlama with LoRA on A100
```shell
python3 train_demo.py --mode inter --encoder_family llama \
  --encoder_name meta-llama/Llama-3.1-8B \
  --use_lora --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --use_bf16 --gradient_checkpointing \
  --lr 1e-4 --batch_size 8 --trainN 5 --N 5 --K 1 --Q 1 \
  --train_iter 600 --val_iter 100 --test_iter 500 --val_step 20 \
  --max_length 64 --model proto
```

#### ProtoLlama Fine-tuning (no LoRA, requires more memory)
```shell
python3 train_demo.py --mode inter --encoder_family llama \
  --encoder_name meta-llama/Llama-3.1-8B \
  --use_bf16 --gradient_checkpointing \
  --lr 1e-5 --batch_size 4 --trainN 5 --N 5 --K 1 --Q 1 \
  --train_iter 600 --val_iter 100 --test_iter 500 --val_step 20 \
  --max_length 64 --model proto
```

#### Test-only (load checkpoint)
```shell
python3 train_demo.py --mode inter --encoder_family llama \
  --encoder_name meta-llama/Llama-3.1-8B \
  --use_lora --load_ckpt checkpoint/proto-inter-5-1-seed0.pth.tar \
  --only_test --test_iter 500 --max_length 64 --model proto
```

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

-- encoder_family       bert or llama (default: bert)
-- encoder_name         HF model id for encoder (e.g. meta-llama/Llama-3.1-8B)
-- tokenizer_name       HF tokenizer id (defaults to encoder_name)
-- use_lora             Enable LoRA adapters for encoder (Llama recommended)
-- lora_r               LoRA rank (default: 16)
-- lora_alpha           LoRA alpha scaling (default: 32)
-- lora_dropout         LoRA dropout (default: 0.05)
-- lora_target_modules  Comma-separated LoRA targets (default: q_proj,k_proj,v_proj,o_proj)
-- use_bf16             Use bfloat16 precision (A100 / newer GPUs)
-- use_8bit             Use 8-bit quantization (requires bitsandbytes)
-- use_4bit             Use 4-bit quantization (requires bitsandbytes)
-- gradient_checkpointing   Enable gradient checkpointing (saves memory, slower)
-- debug_alignment      Print token-label alignment diagnostics (first 5 episodes)
```