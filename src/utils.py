from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class CMConfig():
    output_attention: Optional[bool] = False
    output_attention_norm: Optional[bool] = False
    output_alti: Optional[bool] = False
    output_globenc: Optional[bool] = False
    output_value_zeroing: Optional[bool] = False


def normalize(S):
    return S / S.sum(axis=-1, keepdims=True)

def normalize_(S):
    sums = np.sum(S, axis=-1, keepdims=True)
    mask = np.all(sums == 0, axis=-1, keepdims=True)
    return np.divide(S, sums, out=np.zeros_like(S), where=~mask)


def rollout(S, res=True):
    if res:
        residual_att = np.eye(S.shape[1])[None,...]
        S = S + residual_att
        S = S / S.sum(axis=-1)[...,None]
    
    joint_scores = np.zeros(S.shape)
    layers = joint_scores.shape[0]
    joint_scores[0] = S[0]
    for i in np.arange(1, layers):
        joint_scores[i] = S[i].dot(joint_scores[i-1])
        
    return joint_scores

# for speech encoders
def get_encoder_word_boundaries(start, end, total_enc_frame, total_audio_time):
    start = total_enc_frame * start / total_audio_time
    end = total_enc_frame * end / total_audio_time
    start = np.ceil(start).astype('int')
    end = np.ceil(end).astype('int')
    return start, end