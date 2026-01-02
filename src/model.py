import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from .config import TrainConfig


def _get_neftune_hook(noise_alpha: float):
    def neftune_forward_hook(module, args, output):
        if module.training:
            dims = torch.tensor(output.size(1) * output.size(2))
            mag_norm = noise_alpha / torch.sqrt(dims)
            noise = torch.zeros_like(output).uniform_(-mag_norm, mag_norm)
            return output + noise
        return output

    return neftune_forward_hook


def load_model_and_tokenizer(cfg: TrainConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if "<|im_end|>" in tokenizer.get_vocab():
        tokenizer.eos_token = "<|im_end|>"
    tokenizer.padding_side = "right"

    # Get the local rank for this process to load model directly onto the correct GPU
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}"

    attn_impl = "flash_attention_2" if cfg.use_flash_attn else "eager"
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=attn_impl,
        device_map={"": device}  # Load directly to the correct GPU
    )
    model.config.use_cache = False

    if cfg.neftune_noise_alpha > 0:
        hook_fn = _get_neftune_hook(cfg.neftune_noise_alpha)
        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            model.model.embed_tokens.register_forward_hook(hook_fn)
        elif hasattr(model, "embed_tokens"):
            model.embed_tokens.register_forward_hook(hook_fn)

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, tokenizer