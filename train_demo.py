from transformers import AutoTokenizer
from util.data_loader import get_loader
from util.framework import FewShotNERFramework
from util.word_encoder import BERTWordEncoder, LlamaWordEncoder, PEFT_AVAILABLE
from model.proto import Proto
import sys
import torch
from torch import optim, nn
import numpy as np
import json
import argparse
import os
import torch
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def resolve_encoder_and_tokenizer_args(opt):
    # Keep backward compatibility with existing BERT commands.
    if opt.encoder_family == 'bert':
        encoder_name = opt.encoder_name or opt.pretrain_ckpt or 'bert-base-uncased'
        tokenizer_name = opt.tokenizer_name or encoder_name
    elif opt.encoder_family == 'llama':
        encoder_name = opt.encoder_name or opt.pretrain_ckpt
        if encoder_name is None:
            raise ValueError(
                "For --encoder_family llama, provide --encoder_name (or --pretrain_ckpt)."
            )
        tokenizer_name = opt.tokenizer_name or encoder_name
    else:
        raise ValueError(f"Unsupported encoder_family: {opt.encoder_family}")

    if opt.use_lora and opt.encoder_family == 'bert':
        print("Warning: --use_lora is scaffolded but not active for BERT.")

    opt.encoder_name_resolved = encoder_name
    opt.tokenizer_name_resolved = tokenizer_name
    opt.lora_target_modules_list = [
        x.strip() for x in opt.lora_target_modules.split(',') if x.strip()
    ]
    return opt


def load_tokenizer(tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.padding_side = 'right'
    return tokenizer

def main():
    parser = argparse.ArgumentParser()

    # encoder and tokenizer selection (Phase 1 scaffold)
    parser.add_argument('--encoder_family', type=str, default='bert',
          choices=['bert', 'llama'],
          help='Encoder family. Phase 1 supports bert runtime; llama is scaffolded for upcoming phases.')
    parser.add_argument('--encoder_name', type=str, default=None,
          help='HF model id for encoder. If omitted, falls back to --pretrain_ckpt (or bert-base-uncased for bert).')
    parser.add_argument('--tokenizer_name', type=str, default=None,
          help='HF tokenizer id. If omitted, uses resolved encoder_name.')

    # LoRA scaffold (Phase 1: args only, no runtime wiring yet)
    parser.add_argument('--use_lora', action='store_true',
          help='Enable LoRA config (runtime wiring comes in Phase 3).')
    parser.add_argument('--lora_r', type=int, default=16,
          help='LoRA rank.')
    parser.add_argument('--lora_alpha', type=int, default=32,
          help='LoRA alpha scaling.')
    parser.add_argument('--lora_dropout', type=float, default=0.05,
          help='LoRA dropout.')
    parser.add_argument('--lora_target_modules', type=str, default='q_proj,k_proj,v_proj,o_proj',
          help='Comma-separated target module names for LoRA.')
    parser.add_argument('--lora_bias', type=str, default='none',
          choices=['none', 'all', 'lora_only'],
          help='Bias strategy for LoRA.')
    
    parser.add_argument('--mode', default='inter',
            help='training mode, must be in [inter, intra]')
    parser.add_argument('--trainN', default=2, type=int,
            help='N in train')
    parser.add_argument('--N', default=2, type=int,
            help='N way')
    parser.add_argument('--K', default=2, type=int,
            help='K shot')
    parser.add_argument('--Q', default=3, type=int,
            help='Num of query per class')
    parser.add_argument('--batch_size', default=4, type=int,
            help='batch size')
    parser.add_argument('--train_iter', default=600, type=int,
            help='num of iters in training')
    parser.add_argument('--val_iter', default=100, type=int,
            help='num of iters in validation')
    parser.add_argument('--test_iter', default=500, type=int,
            help='num of iters in testing')
    parser.add_argument('--val_step', default=20, type=int,
           help='val after training how many iters')
    parser.add_argument('--model', default='proto',
            help='model name, must be proto, nnshot, or structshot')
    parser.add_argument('--max_length', default=100, type=int,
           help='max length')
    parser.add_argument('--metrics_dir', type=str, default='metrics',
        help='Directory to save per-run metrics CSVs.')
    parser.add_argument('--lr', default=1e-4, type=float,
           help='learning rate')
    parser.add_argument('--grad_iter', default=1, type=int,
           help='accumulate gradient every x iterations')
    parser.add_argument('--load_ckpt', default=None,
           help='load ckpt')
    parser.add_argument('--save_ckpt', default=None,
           help='save ckpt')
    parser.add_argument('--fp16', action='store_true',
           help='use nvidia apex fp16')
    parser.add_argument('--only_test', action='store_true',
           help='only test')
    parser.add_argument('--ckpt_name', type=str, default='',
           help='checkpoint name.')
    parser.add_argument('--seed', type=int, default=0,
           help='random seed')
    parser.add_argument('--ignore_index', type=int, default=-1,
           help='label index to ignore when calculating loss and metrics')
    parser.add_argument('--use_sampled_data', action='store_true',
           help='use released sampled data, the data should be stored at "data/episode-data/" ')
    parser.add_argument('--debug_alignment', action='store_true',
           help='Enable debug printing for token-label alignment checks (first 5 episodes only).')
    
    # Runtime efficiency flags (Phase 3)
    parser.add_argument('--use_bf16', action='store_true',
           help='Use bfloat16 mixed precision (requires GPU support).')
    parser.add_argument('--use_8bit', action='store_true',
           help='Use 8-bit quantization for encoder (requires bitsandbytes).')
    parser.add_argument('--use_4bit', action='store_true',
           help='Use 4-bit quantization for encoder (requires bitsandbytes).')
    parser.add_argument('--gradient_checkpointing', action='store_true',
           help='Enable gradient checkpointing to reduce memory usage (slower training).')
    parser.add_argument('--profile_batches', action='store_true',
           help='Print per-batch timing for train, validation, and test loops.')
    parser.add_argument('--profile_every', type=int, default=1,
           help='When --profile_batches is set, print every N batches.')

    # only for bert / roberta
    parser.add_argument('--pretrain_ckpt', default=None,
           help='bert / roberta pre-trained checkpoint')

    # only for prototypical networks
    parser.add_argument('--dot', action='store_true', 
           help='use dot instead of L2 distance for proto')

    # only for structshot
    parser.add_argument('--tau', default=0.05, type=float,
           help='StructShot parameter to re-normalizes the transition probabilities')

    # experiment
    parser.add_argument('--use_sgd_for_bert', action='store_true',
           help='use SGD instead of AdamW for BERT.')

    opt = parser.parse_args()
    if opt.profile_every < 1:
        raise ValueError("--profile_every must be >= 1")
    opt = resolve_encoder_and_tokenizer_args(opt)

    if opt.encoder_family == 'llama' and (opt.use_4bit or opt.use_8bit):
        raise NotImplementedError(
            "--use_4bit/--use_8bit flags are parsed but quantized loading is not wired yet in this branch. "
            "Run without these flags, or implement BitsAndBytesConfig-based loading first."
        )

    if opt.use_lora and not PEFT_AVAILABLE:
        raise ImportError(
            "--use_lora was requested but PEFT is not installed in the active environment. "
            "Install with: pip install peft"
        )

    trainN = opt.trainN
    N = opt.N
    K = opt.K
    Q = opt.Q
    batch_size = opt.batch_size
    model_name = opt.model
    max_length = opt.max_length

    print("{}-way-{}-shot Few-Shot NER".format(N, K))
    print("model: {}".format(model_name))
    print("max_length: {}".format(max_length))
    print('mode: {}'.format(opt.mode))
    print('encoder_family: {}'.format(opt.encoder_family))
    print('encoder_name: {}'.format(opt.encoder_name_resolved))
    print('tokenizer_name: {}'.format(opt.tokenizer_name_resolved))
    print('use_lora: {}'.format(opt.use_lora))
    if opt.use_lora:
        print('lora_r: {}, lora_alpha: {}, lora_dropout: {}, lora_bias: {}, lora_targets: {}'.format(
            opt.lora_r, opt.lora_alpha, opt.lora_dropout, opt.lora_bias, ','.join(opt.lora_target_modules_list)
        ))

    set_seed(opt.seed)
    print('loading model and tokenizer...')
    if opt.encoder_family == 'bert':
        pretrain_ckpt = opt.encoder_name_resolved
        word_encoder = BERTWordEncoder(pretrain_ckpt)
        tokenizer = load_tokenizer(opt.tokenizer_name_resolved)
    elif opt.encoder_family == 'llama':
        pretrain_ckpt = opt.encoder_name_resolved
        lora_config = {
            'lora_r': opt.lora_r,
            'lora_alpha': opt.lora_alpha,
            'lora_dropout': opt.lora_dropout,
            'lora_target_modules': opt.lora_target_modules_list,
            'lora_bias': opt.lora_bias
        }
        word_encoder = LlamaWordEncoder(pretrain_ckpt, use_lora=opt.use_lora, lora_config=lora_config)
        tokenizer = load_tokenizer(opt.tokenizer_name_resolved)
        
        if opt.gradient_checkpointing:
            if hasattr(word_encoder.model, "enable_input_require_grads"):
                word_encoder.model.enable_input_require_grads()
            elif hasattr(word_encoder.model, "get_input_embeddings"):
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                word_encoder.model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            word_encoder.model.gradient_checkpointing_enable()
        
        if opt.use_bf16:
            print("[INFO] Using bfloat16 precision for Llama encoder")
            word_encoder.model = word_encoder.model.to(torch.bfloat16)
    else:
        raise ValueError("Unsupported encoder_family: {}".format(opt.encoder_family))

    print('loading data...')
    if not opt.use_sampled_data:
        opt.train = f'data/{opt.mode}/train.txt'
        opt.test = f'data/{opt.mode}/test.txt'
        opt.dev = f'data/{opt.mode}/dev.txt'
        if not (os.path.exists(opt.train) and os.path.exists(opt.dev) and os.path.exists(opt.test)):
            os.system(f'bash data/download.sh {opt.mode}')
    else:
        opt.train = f'data/episode-data/{opt.mode}/train_{opt.N}_{opt.K}.jsonl'
        opt.test = f'data/episode-data/{opt.mode}/test_{opt.N}_{opt.K}.jsonl'
        opt.dev = f'data/episode-data/{opt.mode}/dev_{opt.N}_{opt.K}.jsonl'
        if not (os.path.exists(opt.train) and os.path.exists(opt.dev) and os.path.exists(opt.test)):
            os.system(f'bash data/download.sh episode-data')
            os.system('unzip -d data/ data/episode-data.zip')
    
    if opt.mode == "supervised":
        print("Warning: you are running few-shot learning methods on `supervised` dataset, if it is not expected, please change to `--mode inter` or `--mode intra`.")

    train_data_loader = get_loader(opt.train, tokenizer,
            N=trainN, K=K, Q=Q, batch_size=batch_size, max_length=max_length, ignore_index=opt.ignore_index, use_sampled_data=opt.use_sampled_data, debug_alignment=opt.debug_alignment)
    val_data_loader = get_loader(opt.dev, tokenizer,
            N=N, K=K, Q=Q, batch_size=batch_size, max_length=max_length, ignore_index=opt.ignore_index, use_sampled_data=opt.use_sampled_data, debug_alignment=opt.debug_alignment)
    test_data_loader = get_loader(opt.test, tokenizer,
            N=N, K=K, Q=Q, batch_size=batch_size, max_length=max_length, ignore_index=opt.ignore_index, use_sampled_data=opt.use_sampled_data, debug_alignment=opt.debug_alignment)

        
    prefix = '-'.join([model_name, opt.mode, str(N), str(K), 'seed'+str(opt.seed)])
    if opt.dot:
        prefix += '-dot'
    if len(opt.ckpt_name) > 0:
        prefix += '-' + opt.ckpt_name
    
    if model_name == 'proto':
        print('use proto')
        model = Proto(word_encoder, dot=opt.dot, ignore_index=opt.ignore_index)
        framework = FewShotNERFramework(train_data_loader, val_data_loader, test_data_loader, use_sampled_data=opt.use_sampled_data)
    elif model_name == 'nnshot':
        print('use nnshot')
        model = NNShot(word_encoder, dot=opt.dot, ignore_index=opt.ignore_index)
        framework = FewShotNERFramework(train_data_loader, val_data_loader, test_data_loader, use_sampled_data=opt.use_sampled_data)
    elif model_name == 'structshot':
        print('use structshot')
        model = NNShot(word_encoder, dot=opt.dot, ignore_index=opt.ignore_index)
        framework = FewShotNERFramework(train_data_loader, val_data_loader, test_data_loader, N=opt.N, tau=opt.tau, train_fname=opt.train, viterbi=True, use_sampled_data=opt.use_sampled_data)
    else:
        raise NotImplementedError

    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    ckpt = 'checkpoint/{}.pth.tar'.format(prefix)
    if opt.save_ckpt:
        ckpt = opt.save_ckpt
    print('model-save-path:', ckpt)

    if torch.cuda.is_available():
        model.cuda()

    if not opt.only_test:
        if opt.lr == -1:
            opt.lr = 2e-5

        framework.train(model, prefix,
                load_ckpt=opt.load_ckpt, save_ckpt=ckpt,
                val_step=opt.val_step, fp16=opt.fp16,
                train_iter=opt.train_iter, warmup_step=int(opt.train_iter * 0.1), val_iter=opt.val_iter, learning_rate=opt.lr, use_sgd_for_bert=opt.use_sgd_for_bert,
                profile_batches=opt.profile_batches, profile_every=opt.profile_every)
    else:
        ckpt = opt.load_ckpt
        if ckpt is None:
            print("Warning: --load_ckpt is not specified. Will load Hugginface pre-trained checkpoint.")
            ckpt = 'none'

    # test
    if not os.path.exists(opt.metrics_dir):
        os.makedirs(opt.metrics_dir, exist_ok=True)
    metrics_csv = os.path.join(opt.metrics_dir, f"{prefix}.csv")
    precision, recall, f1, fp, fn, within, outer = framework.eval(
        model,
        opt.test_iter,
        ckpt=ckpt,
        save_csv=metrics_csv,
        profile_batches=opt.profile_batches,
        profile_every=opt.profile_every,
    )
    print("RESULT: precision: %.4f, recall: %.4f, f1:%.4f" % (precision, recall, f1))
    print('ERROR ANALYSIS: fp: %.4f, fn: %.4f, within:%.4f, outer: %.4f'%(fp, fn, within, outer))

if __name__ == "__main__":
    main()
