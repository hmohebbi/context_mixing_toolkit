# A handy library for measuring context-mixing in Transformers
---

## Measures of context-mixing:

- **Attention:** Raw self-attention weights averaged over all heads

- **Attention-Rollout:** Aggregated attention weights over previous layers using Rollout method ([Abnar & Zuidema, ACL 2020](https://aclanthology.org/2020.acl-main.385.pdf))

- **Attention-Norm:** Norm of multiplication of attention weights and transformed value vectors ([Kobayashi et al., EMNLP 2020](https://aclanthology.org/2020.emnlp-main.574.pdf))
  
- **Attention-Norm + RES1:** Incorporates the effect of the first residual stream into the Attention-Norm ([Kobayashi et al., EMNLP 2021](https://aclanthology.org/2021.emnlp-main.373.pdf))

- **Attention-Norm + RES1 + LN1:** Incorporates the effect of the first residual stream and layer normalization into the Attention-Norm ([Kobayashi et al., EMNLP 2021](https://aclanthology.org/2021.emnlp-main.373.pdf))

- **GlobEnc:** Rollout version of Attention-Norm + RES1 + LN1 where the effect of the second layer normalization is also taken into account ([Modarressi et al., NAACL 2022](https://aclanthology.org/2022.naacl-main.19.pdf))

- **Value Zeroing:** Considers all components inside Transformer by measuring how much token representations are affected when nullifying the value vector of each token ([Mohebbi et al., EACL 2023](https://aclanthology.org/2023.eacl-main.245.pdf))

- Other methods not implemented in this repo: LRP-based Attention ([Chefer et al. CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Chefer_Transformer_Interpretability_Beyond_Attention_Visualization_CVPR_2021_paper.pdf)), HTA ([Brunner et al., ICLR 2020](https://arxiv.org/pdf/1908.04211)), ALTI ([Ferrando et al., EMNLP 2022](https://aclanthology.org/2022.emnlp-main.595.pdf))


## How to use?
```python
INPUT_EXAMPLE = "Either you win the game or you"
cm_config = CMConfig(output_attention=True, output_attention_norm=True, output_globenc=True, output_value_zeroing=True)
inputs = tokenizer(INPUT_EXAMPLE, return_tensors="pt")
with torch.no_grad():
  outputs = model(**inputs, output_context_mixings=cm_config)
```

## Notebooks
Colab notebooks are available for both [text](https://colab.research.google.com/drive/114YigbeMilvetmPStnlYR7Wd7gxWYFAX) and [speech](https://colab.research.google.com/drive/1SbRsqU52tGKU3-N_469KCZ-PtRixanE2?usp=sharing) Transformer models.
