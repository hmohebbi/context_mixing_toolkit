# A tiny library for measuring context-mixing in Transformers
---

## Measures of context-mixing:

- **Attention:** Raw self-attention weights averaged over all heads

- **Attention-Rollout:** Aggregated attention weights over previous layers using Rollout method (Abnar & Zuidema, ACL 2020)

- **Attention-Norm:** Norm of multiplication of attention weights and transformed value vectors (Kobayashi et al., EMNLP 2020)
  
- **Attention-Norm + RES1:** Incorporates the effect of the first residual stream into the Attention-Norm (Kobayashi et al., EMNLP 2021)

- **Attention-Norm + RES1 + LN1:** Incorporates the effect of the first residual stream and layer normalization into the Attention-Norm (Kobayashi et al., EMNLP 2021)

- **GlobEnc:** Rollout version of Attention-Norm + RES1 + LN1 where the effect of the second layer normalization is also taken into account (Modarressi et al., NAACL 2022)

- **Value Zeroing:** Considers all components inside Transformer by measuring how much token representations are affected when nullifying the value vector of each token (Mohebbi et al., EACL 2023)

- Other methods not implemented in this repo: LRP-based Attention (Chefer et al. CVPR 2021), HTA (Brunner et al., ICLR 2020), ALTI (Ferrando et al., EMNLP 2022)


## How to use?
[[Colab notebook](https://colab.research.google.com/drive/114YigbeMilvetmPStnlYR7Wd7gxWYFAX)]
```
INPUT_EXAMPLE = "Either you win the game or you"
cm_config = CMConfig(output_attention=True, output_attention_norm=True, output_globenc=True, output_value_zeroing=True)
inputs = tokenizer(INPUT_EXAMPLE, return_tensors="pt")
with torch.no_grad():
  outputs = model(**inputs, output_context_mixings=cm_config)
```

