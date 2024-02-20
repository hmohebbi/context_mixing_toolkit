from dataclasses import dataclass
from typing import Optional


@dataclass
class CMConfig():
    output_attention: Optional[bool] = False
    output_attention_norm: Optional[bool] = False
    output_alti: Optional[bool] = False
    output_globenc: Optional[bool] = False
    output_value_zeroing: Optional[bool] = False