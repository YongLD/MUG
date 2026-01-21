from vlmeval.vlm import *
from vlmeval.api import *
from functools import partial

LLAVA_V1_7B_MODEL_PTH = 'Please set your local path to LLaVA-7B-v1.1 here, the model weight is obtained by merging LLaVA delta weight based on vicuna-7b-v1.1 in https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md with vicuna-7b-v1.1. '


qwen2vl_series = {
    "Qwen2.5-VL-3B-Instruct": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2.5-VL-3B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    "Qwen2.5-VL-7B-Instruct": partial(
        Qwen2VLChat,
        model_path="/extrahome0/ldy/model/Qwen2.5-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),

    "Qwen2.5-VL-32B-Instruct": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2.5-VL-32B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
#         use_vllm=True
    ),
    "Qwen2.5-VL-72B-Instruct": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2.5-VL-72B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    
}

supported_VLM = {}

model_groups = [
    qwen2vl_series,
]

for grp in model_groups:
    supported_VLM.update(grp)
