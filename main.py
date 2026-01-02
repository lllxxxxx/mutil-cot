import sys
import os
import gc
import torch
from torch.utils.data import DataLoader
from transformers import HfArgumentParser

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import TrainConfig
from src.utils import setup_seed
from src.data import QwenSFTDataset, QwenPredictDataset, ManualCollator
from src.model import load_model_and_tokenizer
from src.trainer import ManualTrainer


def main():
    # Clear GPU cache at start
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    parser = HfArgumentParser((TrainConfig,))
    cfg, = parser.parse_args_into_dataclasses()

    setup_seed(cfg.seed)
    model, tokenizer = load_model_and_tokenizer(cfg)

    # Create datasets
    train_dataset = QwenSFTDataset(cfg.read_data_paths, tokenizer, cfg.max_length, cfg.seed)

    val_datasets = []
    test_datasets = []
    for view in cfg.view_names:
        val_datasets.append(QwenPredictDataset(cfg.val_data_path, tokenizer, cfg.max_length, view_name=view))
        test_datasets.append(QwenPredictDataset(cfg.test_data_path, tokenizer, cfg.max_length, view_name=view))

    # Create collators
    train_collator = ManualCollator(tokenizer, use_flash_attn=cfg.use_flash_attn, for_inference=False,
                                    use_4d_mask=cfg.enable_4d_mask)
    infer_collator = ManualCollator(tokenizer, for_inference=True)

    # Pass datasets to trainer - it will create dataloaders with proper distributed setup
    trainer = ManualTrainer(
        cfg, model, tokenizer,
        train_dataset=train_dataset,
        train_collator=train_collator,
        val_datasets=val_datasets,
        test_datasets=test_datasets,
        infer_collator=infer_collator
    )

    trainer.train()


if __name__ == "__main__":
    main()