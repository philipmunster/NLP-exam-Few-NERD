import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
from torch import optim
from transformers import AutoModel, BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification, RobertaModel, RobertaTokenizer, RobertaForSequenceClassification

try:
    from transformers import BitsAndBytesConfig
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BitsAndBytesConfig = None
    BITSANDBYTES_AVAILABLE = False

try:
    from peft import get_peft_model, LoraConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

try:
    from peft import prepare_model_for_kbit_training
except ImportError:
    prepare_model_for_kbit_training = None

def apply_lora_to_model(model, lora_r=16, lora_alpha=32, lora_dropout=0.05, lora_target_modules=None, lora_bias='none', is_quantized=False):
    """
    Apply PEFT LoRA adapters to a model. Returns the LoRA-wrapped model.
    """
    if not PEFT_AVAILABLE:
        print("[WARNING] PEFT not installed. LoRA will be skipped. Install with: pip install peft")
        return model

    if is_quantized:
        if prepare_model_for_kbit_training is None:
            raise ImportError("PEFT prepare_model_for_kbit_training is required for k-bit LoRA training.")
        model = prepare_model_for_kbit_training(model)
    
    if lora_target_modules is None:
        lora_target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias=lora_bias,
        task_type='FEATURE_EXTRACTION'
    )
    model = get_peft_model(model, lora_config)
    print(f"[INFO] Applied LoRA to model: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    return model

class BERTWordEncoder(nn.Module):

    def __init__(self, pretrain_path): 
        nn.Module.__init__(self)
        self.bert = BertModel.from_pretrained(pretrain_path)

    def forward(self, words, masks):
        outputs = self.bert(words, attention_mask=masks, output_hidden_states=True, return_dict=True)
        #outputs = self.bert(inputs['word'], attention_mask=inputs['mask'], output_hidden_states=True, return_dict=True)
        # use the sum of the last 4 layers
        last_four_hidden_states = torch.cat([hidden_state.unsqueeze(0) for hidden_state in outputs['hidden_states'][-4:]], 0)
        del outputs
        word_embeddings = torch.sum(last_four_hidden_states, 0) # [num_sent, number_of_tokens, 768]
        return word_embeddings


class LlamaWordEncoder(nn.Module):
    def __init__(self, pretrain_path, aggregation="last", use_lora=False, lora_config=None, use_4bit=False, use_8bit=False):
        super().__init__()
        if use_4bit and use_8bit:
            raise ValueError("Use only one of use_4bit or use_8bit.")

        model_kwargs = {}
        self.is_quantized = use_4bit or use_8bit
        if use_4bit:
            if not BITSANDBYTES_AVAILABLE:
                raise ImportError("4-bit loading requires bitsandbytes support in transformers.")
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["device_map"] = "auto"
            print("[INFO] Loading Llama encoder in 4-bit NF4")
        elif use_8bit:
            if not BITSANDBYTES_AVAILABLE:
                raise ImportError("8-bit loading requires bitsandbytes support in transformers.")
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            model_kwargs["device_map"] = "auto"
            print("[INFO] Loading Llama encoder in 8-bit")

        self.model = AutoModel.from_pretrained(pretrain_path, **model_kwargs)
        self.aggregation = aggregation
        self.use_lora = use_lora
        
        if use_lora:
            if lora_config is None:
                lora_config = {}
            self.model = apply_lora_to_model(self.model, is_quantized=self.is_quantized, **lora_config)

    def forward(self, words, masks):
        outputs = self.model(
            input_ids=words,
            attention_mask=masks,
            output_hidden_states=True,
            return_dict=True,
        )

        if self.aggregation == "last":
            word_embeddings = outputs.last_hidden_state
        elif self.aggregation == "last4_sum":
            last_four = outputs.hidden_states[-4:]
            word_embeddings = torch.stack(last_four, dim=0).sum(dim=0)
        elif self.aggregation == "last4_mean":
            last_four = outputs.hidden_states[-4:]
            word_embeddings = torch.stack(last_four, dim=0).mean(dim=0)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        return word_embeddings
