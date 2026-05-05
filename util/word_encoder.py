import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
from torch import optim
from transformers import AutoModel, BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification, RobertaModel, RobertaTokenizer, RobertaForSequenceClassification

try:
    from peft import get_peft_model, LoraConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

def apply_lora_to_model(model, lora_r=16, lora_alpha=32, lora_dropout=0.05, lora_target_modules=None, lora_bias='none'):
    """
    Apply PEFT LoRA adapters to a model. Returns the LoRA-wrapped model.
    """
    if not PEFT_AVAILABLE:
        print("[WARNING] PEFT not installed. LoRA will be skipped. Install with: pip install peft")
        return model
    
    if lora_target_modules is None:
        lora_target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias=lora_bias,
        task_type='CAUSAL_LM'
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
    def __init__(self, pretrain_path, aggregation="last", use_lora=False, lora_config=None):
        super().__init__()
        self.model = AutoModel.from_pretrained(pretrain_path)
        self.aggregation = aggregation
        self.use_lora = use_lora
        
        if use_lora:
            if lora_config is None:
                lora_config = {}
            self.model = apply_lora_to_model(self.model, **lora_config)

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