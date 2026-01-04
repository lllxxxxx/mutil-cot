from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class TrainConfig:
    seed: int = 42
    model_name_or_path: str = "/root/autodl-tmp/Qwen2.5-7B"
    template: str = field(default="qwen", metadata={"help": "使用的对话模版类型"})

    # --- 路径配置 ---
    raw_data_path: str = "./data/raw/train_data.json"
    val_data_path: str = "./data/raw/dev_data.json"
    test_data_path: str = "./data/raw/test_data.json"
    output_dir: str = "./saves/qwen2.5-7B"
    num_gpus: int = 2 # Default to 2 GPUs, adjust as needed

    # --- API 与 数据生成配置 ---
    api_key: str = "sk-9b269e0b1d8d410f9bcd8373e48c0842"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_model: str = "qwen-flash"
    api_concurrency: int = 10

    prompt_files: List[str] = field(default_factory=lambda: [
        "./data/prompts/view1.txt",
        "./data/prompts/view2.txt",
        "./data/prompts/view3.txt"
    ])

    view_names: List[str] = field(default_factory=lambda: [
        "指代消解", "相关对象", "表情包与梗"
    ])

    generated_data_paths: List[str] = field(default_factory=lambda: [
        "./data/processed/train_view1.json",
        "./data/processed/train_view2.json",
        "./data/processed/train_view3.json"
    ])

    read_data_paths: List[str] = field(default_factory=lambda: [
        "./data/processed/train_view1.json",
        "./data/processed/train_view2.json",
        "./data/processed/train_view3.json",
        "./data/processed/train_raw_view.json"
    ])

    # --- 训练超参 ---
    max_length: int = 2048
    batch_size: int = 4  # Per GPU batch size
    gen_batch_size: int = 16  # Per GPU inference batch size
    gradient_accumulation_steps: int = 8  # Effective batch = batch_size * num_gpus * grad_accum
    learning_rate: float = 3e-5
    num_epochs: int = 3
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    eval_ratio: float = 0.25

    # --- LoRA ---
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])

    # --- 其他 ---
    use_swanlab: bool = True
    swanlab_project: str = "4-mix"
    swanlab_run_name: str = "multi_view_cot"
    swanlab_api_key: Optional[str] = "EcJUnP1993IKCvYXbXxJo"
    logging_steps: int = 5
    save_steps: int = 1000

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_flash_attn: bool = True
    enable_4d_mask: bool = False
    neftune_noise_alpha: float = 5.0