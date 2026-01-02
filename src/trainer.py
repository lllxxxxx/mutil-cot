import os
import torch
import re
import collections
from tqdm import tqdm
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
import swanlab
from transformers import logging
logging.set_verbosity_error()

SWANLAB_INSTALLED = True


class ManualTrainer:
    def __init__(self, cfg, model, tokenizer, train_dataloader, val_dataloaders=None, test_dataloaders=None):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader

        self.val_dataloaders = val_dataloaders
        self.test_dataloaders = test_dataloaders

        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.best_metric = -float('inf')

        os.makedirs(self.cfg.output_dir, exist_ok=True)
        if self.cfg.use_swanlab:
            self._init_tracker()

    def _create_optimizer(self):
        decay_params = [n for n, p in self.model.named_parameters() if "bias" not in n and "norm" not in n]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if n in decay_params and p.requires_grad],
             "weight_decay": self.cfg.weight_decay},
            {"params": [p for n, p in self.model.named_parameters() if n not in decay_params and p.requires_grad],
             "weight_decay": 0.0},
        ]
        return AdamW(optimizer_grouped_parameters, lr=self.cfg.learning_rate)

    def _create_scheduler(self):
        self.total_steps = len(self.train_dataloader) * self.cfg.num_epochs // self.cfg.gradient_accumulation_steps
        return get_cosine_schedule_with_warmup(self.optimizer,
                                               num_warmup_steps=int(self.total_steps * self.cfg.warmup_ratio),
                                               num_training_steps=self.total_steps)

    def _init_tracker(self):
        if self.cfg.swanlab_api_key: swanlab.login(api_key=self.cfg.swanlab_api_key)
        swanlab.init(project=self.cfg.swanlab_project, experiment_name=self.cfg.swanlab_run_name,
                     config=self.cfg.__dict__, mode="cloud")

    def train(self):
        self.model.train()
        global_step = 0
        current_loss = 0.0
        progress_bar = tqdm(total=self.total_steps, desc="Training", unit="step")

        eval_step=int(len(self.train_dataloader)*self.cfg.eval_ratio/self.cfg.gradient_accumulation_steps)
        for epoch in range(self.cfg.num_epochs):
            epoch_total_loss = 0.0
            for step, batch in enumerate(self.train_dataloader):
                inputs = {k: v.to(self.cfg.device) for k, v in batch.items() if
                          k in ["input_ids", "labels", "attention_mask"] and isinstance(v, torch.Tensor)}
                outputs = self.model(**inputs)

                epoch_total_loss += outputs.loss.item()
                loss = outputs.loss / self.cfg.gradient_accumulation_steps
                loss.backward()
                current_loss += loss.item()

                if (step + 1) % self.cfg.gradient_accumulation_steps == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                    progress_bar.update(1)
                    if global_step % self.cfg.logging_steps == 0:
                        logs = {
                            "train/loss_step": current_loss, "train/grad_norm": grad_norm.item(),
                            "train/lr": self.scheduler.get_last_lr()[0],
                            "train/epoch": epoch + (step + 1) / len(self.train_dataloader)
                        }
                        progress_bar.set_postfix(loss=f"{current_loss:.4f}", gn=f"{grad_norm.item():.2f}")
                        if self.cfg.use_swanlab and SWANLAB_INSTALLED: swanlab.log(logs)
                        current_loss = 0.0
                    # if global_step % self.cfg.save_steps == 0: self.save_checkpoint(f"step-{global_step}")
                    if global_step % eval_step == 0:
                        if self.val_dataloaders:
                            val_score=self.evaluate(self.val_dataloaders, epoch=global_step/eval_step, stage="val")
                            if val_score > self.best_metric:
                                self.best_metric = val_score
                                print(f" >> New Best Val F1: {self.best_metric:.4f} (Saving...)")
                                self.save_checkpoint("best_model")
                                if self.cfg.use_swanlab and SWANLAB_INSTALLED: swanlab.log(
                                    {"val/best_f1": self.best_metric})
                            self.model.train()

            avg_epoch_loss = epoch_total_loss / len(self.train_dataloader)
            print(f"\n=== Epoch {epoch + 1} Avg Loss: {avg_epoch_loss:.4f} ===")
            if self.cfg.use_swanlab and SWANLAB_INSTALLED: swanlab.log(
                {"train/loss_epoch_avg": avg_epoch_loss, "train/epoch": epoch + 1})
            self.model.train()

        progress_bar.close()
        if self.test_dataloaders: self.predict_on_test_set()
        if self.cfg.use_swanlab and SWANLAB_INSTALLED: swanlab.finish()

    def save_checkpoint(self, tag):
        path = os.path.join(self.cfg.output_dir, f"checkpoint-{tag}" if isinstance(tag, int) else str(tag))
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def predict_on_test_set(self):
        print("\n*** Loading Best Model for Testing ***")
        path = os.path.join(self.cfg.output_dir, "best_model")
        if os.path.exists(path): self.model.load_adapter(path, adapter_name="default")
        self.evaluate(self.test_dataloaders, epoch="TEST", stage="test")

    @torch.inference_mode()
    def evaluate(self, dataloaders, epoch, stage="val"):
        """
        接收 List[DataLoader]，依次推理，结果全在内存聚合，不写文件。
        """
        self.model.eval()
        self.tokenizer.padding_side = "left"

        results_map = collections.defaultdict(list)
        gold_labels_map = {}

        for i, loader in enumerate(dataloaders):
            view_name = self.cfg.view_names[i]
            tmp_pred=[]
            tmp_label=[]
            for batch in tqdm(loader, desc=f"{stage} {view_name}", leave=False):
                inputs = {
                    "input_ids": batch["input_ids"].to(self.cfg.device),
                    "attention_mask": batch["attention_mask"].to(self.cfg.device)
                }

                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=168,
                    do_sample=False,
                    eos_token_id=self.tokenizer.eos_token_id
                )

                input_len = inputs["input_ids"].shape[1]
                decoded_texts = self.tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)

                batch_idxs = batch["original_idx"]
                batch_labels = batch["label_text"]

                for idx, text, gold in zip(batch_idxs, decoded_texts, batch_labels):
                    if isinstance(idx, torch.Tensor): idx = idx.item()
                    label = self._extract_label(text)
                    results_map[idx].append(label)
                    gold_labels_map[idx] = gold
                    tmp_pred.append(label)
                    tmp_label.append(gold)
            tmp_f1=self._compute_char_f1(tmp_pred, tmp_label)
            print(f"角度：{view_name}   F1: {tmp_f1:.4f}")
            if self.cfg.use_swanlab and SWANLAB_INSTALLED:
                swanlab.log({f"{view_name}/f1": tmp_f1})

        final_preds = []
        final_truths = []

        sorted_idxs = sorted(results_map.keys())

        for idx in sorted_idxs:
            votes = results_map[idx]
            gold = gold_labels_map[idx]


            if not votes: votes = ["中立"]
            final_vote = collections.Counter(votes).most_common(1)[0][0]

            final_preds.append(final_vote)
            final_truths.append(gold)

        self.tokenizer.padding_side = "right"

        score = self._compute_char_f1(final_preds, final_truths)
        print(f"Epoch {epoch} {stage.capitalize()} F1 Score: {score:.4f}")

        if self.cfg.use_swanlab and SWANLAB_INSTALLED:
            swanlab.log({f"{stage}/f1": score})

        return score

    def _extract_label(self, text):
        labels = ["支持", "反对", "中立"]
        found = [(text.rfind(l), l) for l in labels if l in text]

        if not found:
            return "中立"
        return max(found, key=lambda x: x[0])[1]

    def _compute_char_f1(self, preds, truths):
        labels = ["中立", "支持", "反对"]
        f1_s = []
        for label in labels:
            set_1 = {i for i, k in enumerate(truths) if k == label}
            set_2 = {i for i, k in enumerate(preds) if k.strip() == label}
            set_inter = set_1 & set_2
            len_A, len_P, len_TP = len(set_1), len(set_2), len(set_inter)
            p = len_TP / len_P if len_P > 0 else 0.0
            r = len_TP / len_A if len_A > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            f1_s.append(f1)
        return sum(f1_s) / len(f1_s) if f1_s else 0.0