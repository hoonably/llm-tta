# lora/lora_setup.py
import torch.nn as nn
from lora.dynamic_lora import DynamicLoRALinear


def patch_lora_layers(model: nn.Module, target_modules=["q_proj", "v_proj"], r_max=8, alpha=16):
    """
    HuggingFace LLM 모델에서 지정된 Linear 레이어를 DynamicLoRALinear로 치환
    """
    for name, module in model.named_modules():
        for target in target_modules:
            if name.endswith(target):
                parent = get_parent_module(model, name)
                layer_name = name.split(".")[-1]
                old_layer = getattr(parent, layer_name)

                if isinstance(old_layer, nn.Linear):
                    dynamic_layer = DynamicLoRALinear(
                        in_features=old_layer.in_features,
                        out_features=old_layer.out_features,
                        r_max=r_max,
                        lora_alpha=alpha,
                        bias=old_layer.bias is not None
                    )
                    # copy original frozen weight
                    dynamic_layer.weight.data = old_layer.weight.data.clone()
                    if old_layer.bias is not None:
                        dynamic_layer.bias.data = old_layer.bias.data.clone()

                    setattr(parent, layer_name, dynamic_layer)


def get_parent_module(model, full_name: str):
    """
    'model.decoder.layers.0.self_attn.q_proj' 같은 이름에서
    → 해당 모듈의 parent module을 반환
    """
    names = full_name.split(".")
    parent = model
    for n in names[:-1]:
        parent = getattr(parent, n)
    return parent
