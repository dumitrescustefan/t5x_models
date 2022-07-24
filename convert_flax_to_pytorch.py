import sys
import torch
import numpy as np

import jax
import jax.numpy as jnp

from transformers import AutoTokenizer, FlaxGPTNeoForCausalLM, GPTNeoForCausalLM
from transformers import FlaxT5ForConditionalGeneration,TFT5ForConditionalGeneration, T5ForConditionalGeneration, T5Config, AutoTokenizer


MODEL_PATH_FX = sys.argv[1]
MODEL_PATH_PT = sys.argv[2]

# if model weights are bfloat16 convert to float32
def to_f32(t):
    return jax.tree_map(lambda x: x.astype(jnp.float32) if x.dtype==jnp.bfloat16 else x, t)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH_FX)
tokenizer.pad_token = tokenizer.eos_token

model_fx = FlaxT5ForConditionalGeneration.from_pretrained(MODEL_PATH_FX)

model_pt = T5ForConditionalGeneration.from_pretrained(MODEL_PATH_FX, from_flax=True)
model_pt.save_pretrained(MODEL_PATH_PT)

