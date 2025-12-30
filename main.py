import sys
import os
from torch.utils.data import DataLoader
from transformers import HfArgumentParser

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import TrainConfig
from src.utils import setup_seed
from src.data import QwenSFTDataset, QwenPredictDataset, ManualCollator
from src.model import load_model_and_tokenizer
from src.trainer import ManualTrainer


def main():
    parser = HfArgumentParser((TrainConfig,))
    cfg, = parser.parse_args_into_dataclasses()

    setup_seed(cfg.seed)
    model, tokenizer = load_model_and_tokenizer(cfg)

    train_dataset = QwenSFTDataset(cfg.generated_data_paths, tokenizer, cfg.max_length,cfg.seed)
    train_collator = ManualCollator(tokenizer, use_flash_attn=cfg.use_flash_attn, for_inference=False,
                                    use_4d_mask=cfg.enable_4d_mask)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=train_collator,
                              num_workers=4, pin_memory=True)

    infer_collator = ManualCollator(tokenizer, for_inference=True)  # 推理模式，不做4D mask

    val_loaders = []
    test_loaders = []

    for view in cfg.view_names:
        val_ds = QwenPredictDataset(cfg.val_data_path, tokenizer, cfg.max_length, view_name=view)
        val_dl = DataLoader(val_ds, batch_size=cfg.gen_batch_size, shuffle=False, collate_fn=infer_collator, num_workers=4,
                            pin_memory=True)
        val_loaders.append(val_dl)

        test_ds = QwenPredictDataset(cfg.test_data_path, tokenizer, cfg.max_length, view_name=view)
        test_dl = DataLoader(test_ds, batch_size=cfg.gen_batch_size, shuffle=False, collate_fn=infer_collator,
                             num_workers=4, pin_memory=True)
        test_loaders.append(test_dl)

    trainer = ManualTrainer(
        cfg, model, tokenizer,
        train_dataloader=train_loader,
        val_dataloaders=val_loaders,  # List
        test_dataloaders=test_loaders  # List
    )

    trainer.train()


if __name__ == "__main__":
    main()