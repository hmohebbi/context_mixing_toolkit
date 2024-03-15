
MODEL_PATH = "gpt2" # options: bert-base-uncased, roberta-base, gpt2, google/gemma-2b

if MODEL_PATH.split('-')[0] == "bert":
    INPUT_EXAMPLE = "Either you win the game or you [MASK] the game."
elif MODEL_PATH.split('-')[0] == "roberta":
    INPUT_EXAMPLE = "Either you win the game or you <mask> the game."
elif "gemma" in MODEL_PATH or "gpt2" in MODEL_PATH:
    INPUT_EXAMPLE = "Either you win the game or you"

import pandas as pd
from plotnine import *
from IPython.display import display
import numpy as np
import torch
from transformers import AutoTokenizer
from src.modeling_bert import BertModel
from src.modeling_roberta import RobertaModel
from src.modeling_gemma import GemmaModel
from src.modeling_gpt2 import GPT2Model
from src.utils import CMConfig, normalize, rollout

# from huggingface_hub import notebook_login
# notebook_login()

cm_config = CMConfig(output_attention=True, output_value_zeroing=True, output_attention_norm=False, output_globenc=False)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if "roberta" in MODEL_PATH:
    model = BertModel.from_pretrained(MODEL_PATH)
elif "bert" in MODEL_PATH:
    model = RobertaModel.from_pretrained(MODEL_PATH)
elif "gpt2" in MODEL_PATH:
    model = GPT2Model.from_pretrained(MODEL_PATH)
elif "gemma" in MODEL_PATH:
    model = GemmaModel.from_pretrained(MODEL_PATH, attn_implementation='eager') #, torch_dtype=torch.bfloat16
else:
    raise ValueError("Context mixing methods have not been implemented for this model yet!")
model.eval()

inputs = tokenizer(INPUT_EXAMPLE, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs, output_context_mixings=cm_config)

scores = {}
scores['Attention'] = normalize(torch.stack(outputs['context_mixings']['attention']).permute(1, 0, 2, 3).squeeze(0).detach().cpu().type(torch.float32).numpy())
scores['Value Zeroing'] = normalize(torch.stack(outputs['context_mixings']['value_zeroing']).permute(1, 0, 2, 3).squeeze(0).detach().cpu().type(torch.float32).numpy())
if "roberta" in MODEL_PATH or "bert" in MODEL_PATH:
    scores['Attention-Norm'] = normalize(torch.stack(outputs['context_mixings']['attention_norm']).permute(1, 0, 2, 3).squeeze(0).detach().cpu().type(torch.float32).numpy())
    scores['Attention-Norm + RES1'] = normalize(torch.stack(outputs['context_mixings']['attention_norm_res']).permute(1, 0, 2, 3).squeeze(0).detach().cpu().type(torch.float32).numpy())
    scores['Attention-Norm + RES1 + LN1'] = normalize(torch.stack(outputs['context_mixings']['attention_norm_res_ln']).permute(1, 0, 2, 3).squeeze(0).detach().cpu().type(torch.float32).numpy())
    scores['GlobEnc'] = rollout(normalize(torch.stack(outputs['context_mixings']['globenc']).permute(1, 0, 2, 3).squeeze(0).detach().cpu().type(torch.float32).numpy()), res=False)

# plot
tokens = [tokenizer.convert_ids_to_tokens(t) for t in inputs['input_ids'][0].detach().cpu().numpy().tolist()]
token_orders = list(range(len(tokens)))
order_to_token_mapper = {i: tokens[i] for i in token_orders}

NAMES = list(scores.keys())
num_layers, seq_len, _= scores[NAMES[0]].shape
for l in range(num_layers):
    df_list = []
    for name in NAMES:
        df = pd.DataFrame(scores[name][l], index=token_orders, columns=token_orders).reset_index()
        df = df.melt(id_vars='index')
        df.columns = ['x', 'y', 'value']
        df['method'] = name
        df_list.append(df)
    merged_df = pd.concat(df_list)
    merged_df['x'] = pd.Categorical(merged_df['x'], categories=token_orders)
    merged_df['y'] = pd.Categorical(merged_df['y'], categories=token_orders)

    p = (ggplot(merged_df, aes('y', 'x', fill='value'))
        + geom_tile() 
        + scale_fill_gradient(low='white', high='purple', guide=False)
        + facet_wrap('~method')  
        + theme(axis_text_x=element_text(rotation=90, hjust=1), axis_title_x=element_blank(), axis_title_y=element_blank())
        + scale_x_discrete(labels=[order_to_token_mapper[i] for i in token_orders])
        + scale_y_discrete(labels=[order_to_token_mapper[i] for i in token_orders][::-1], limits=reversed)
        + labs(title=f"L{l+1}")
        )
    display(p)
    