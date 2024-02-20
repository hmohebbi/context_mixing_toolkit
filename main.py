
MODEL_PATH = "bert-base-uncased"

import pandas
import seaborn
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer
from src.modeling_bert import BertModel
from src.cm_utils import CMConfig

def create_plot(all_tokens, scores):
    LAYERS = list(range(12))
    fig, axs = plt.subplots(6, 2, figsize=(8, 24))
    plt.subplots_adjust(top=0.98, bottom=0.05, hspace=0.5, wspace=0.5)
    for layer in LAYERS:
        a = (layer)//2
        b = layer%2
        seaborn.heatmap(
                ax=axs[a, b],
                data=pandas.DataFrame(scores[layer], index= all_tokens, columns=all_tokens),
                cmap="Blues",
                annot=False,
                cbar=False
            )
        axs[a, b].set_title(f"Layer: {layer+1}")
    return fig
    

cm_config = CMConfig(output_attention=True, output_value_zeroing=True, output_attention_norm=True, output_globenc=True) #, output_alti=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = BertModel.from_pretrained(MODEL_PATH)

# inputs = tokenizer("my friend [MASK] fixed this chair.", return_tensors="pt")
inputs = tokenizer(["The pictures of some hat [MASK] scaring Marcus.", "Hi there!"], padding=True, return_tensors="pt")
outputs = model(**inputs, output_context_mixings=cm_config)

attn = torch.stack(outputs['context_mixings']['attention']).permute(1, 0, 2, 3)[0].detach().cpu().numpy()
attn_norm = torch.stack(outputs['context_mixings']['attention_norm']).permute(1, 0, 2, 3)[0].detach().cpu().numpy()
attn_norm_res = torch.stack(outputs['context_mixings']['attention_norm_res']).permute(1, 0, 2, 3)[0].detach().cpu().numpy()
attn_norm_res_ln = torch.stack(outputs['context_mixings']['attention_norm_res_ln']).permute(1, 0, 2, 3)[0].detach().cpu().numpy()
vz = torch.stack(outputs['context_mixings']['value_zeroing']).permute(1, 0, 2, 3)[0].detach().cpu().numpy()
globenc = torch.stack(outputs['context_mixings']['globenc']).permute(1, 0, 2, 3)[0].detach().cpu().numpy()

# plot
scores = globenc
all_tokens = [tokenizer.convert_ids_to_tokens(t) for t in inputs['input_ids'][0].detach().cpu().numpy().tolist()]
fig = create_plot(all_tokens, scores)
fig.show()