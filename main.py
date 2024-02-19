
MODEL_PATH = "bert-base-uncased"


import torch
from transformers import AutoTokenizer
from src.modeling_bert import BertModel
from src.cm_utils import CMConfig

cm_config = CMConfig(output_attention=True, output_value_zeroing=False)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = BertModel.from_pretrained(MODEL_PATH)

inputs = tokenizer("my friend [MASK] fixed this chair.", return_tensors="pt")
outputs = model(**inputs, output_context_mixings=cm_config)

