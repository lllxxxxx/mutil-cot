import os
import torch
import re
import collections
from tqdm import tqdm
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
import swanlab
from transformers import logging
from accelerate import Accelerator
logging.set_verbosity_error()

SWANLAB_INSTALLED = True


class ManualTrainer:
    def __init__(self, cfg, model, tokenizer, train_dataloader, val_dataloaders=None, test_dataloaders=None):
        self.cfg = cfg
        self.accelerator = Accelerator(gradient_accumulation_steps=cfg.gradient_accumulation_steps)
        self.model = model
        self.tokenizer = tokenizer
        
        self.optimizer = self._create_optimizer()
        
        self.train_dataloader, self.val_dataloaders, self.test_dataloaders, self.model, self.optimizer = self.accelerator.prepare(
            train_dataloader, val_dataloaders, test_dataloaders, self.model, self.optimizer
        )
        
   
        self.scheduler = self._create_scheduler() 
        self.scheduler = self.accelerator.prepare(self.scheduler)

        self.best_metric = -float('inf')

        if self.accelerator.is_main_process:
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
        
        if self.accelerator.is_main_process:
            progress_bar = tqdm(total=self.total_steps, desc="Training", unit="step")
        
        eval_step=int(len(self.train_dataloader)*self.cfg.eval_ratio/self.cfg.gradient_accumulation_steps)
        for epoch in range(self.cfg.num_epochs):
            epoch_total_loss = 0.0
            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.model):
                    inputs = {k: v for k, v in batch.items() if
                              k in ["input_ids", "labels", "attention_mask"] and isinstance(v, torch.Tensor)}
                    
                    outputs = self.model(**inputs)

                    epoch_total_loss += outputs.loss.item()
                    loss = outputs.loss 
                    self.accelerator.backward(loss)
                    current_loss += loss.item() / self.cfg.gradient_accumulation_steps

                    if self.accelerator.sync_gradients:
                        grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                        global_step += 1
                        
                        if self.accelerator.is_main_process:
                            progress_bar.update(1)
                    if global_step % self.cfg.logging_steps == 0 and self.accelerator.is_main_process:
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
                            if self.accelerator.is_main_process and val_score > self.best_metric:
                                self.best_metric = val_score
                                print(f" >> New Best Val F1: {self.best_metric:.4f} (Saving...)")
                                self.save_checkpoint("best_model")
                                if self.cfg.use_swanlab and SWANLAB_INSTALLED: swanlab.log(
                                    {"val/best_f1": self.best_metric})
                            self.model.train()

            avg_epoch_loss = epoch_total_loss / len(self.train_dataloader)
            
            self.accelerator.wait_for_everyone()
            
            if self.accelerator.is_main_process:
                print(f"\n=== Epoch {epoch + 1} Avg Loss: {avg_epoch_loss:.4f} ===")
                if self.cfg.use_swanlab and SWANLAB_INSTALLED: swanlab.log(
                    {"train/loss_epoch_avg": avg_epoch_loss, "train/epoch": epoch + 1})
            self.model.train()

        if self.accelerator.is_main_process:
            progress_bar.close()
        
        self.accelerator.wait_for_everyone()
        if self.test_dataloaders: self.predict_on_test_set()
        if self.accelerator.is_main_process and self.cfg.use_swanlab and SWANLAB_INSTALLED: swanlab.finish()

    def save_checkpoint(self, tag):

        path = os.path.join(self.cfg.output_dir, f"checkpoint-{tag}" if isinstance(tag, int) else str(tag))
        os.makedirs(path, exist_ok=True)
        
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def predict_on_test_set(self):

        if self.accelerator.is_main_process:
            print("\n*** Loading Best Model for Testing ***")
        
        path = os.path.join(self.cfg.output_dir, "best_model")
        if os.path.exists(path): 
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.load_adapter(path, adapter_name="default")
            
        self.evaluate(self.test_dataloaders, epoch="TEST", stage="test")

    @torch.inference_mode()
    def evaluate(self, dataloaders, epoch, stage="val"):
        """
        接收 List[DataLoader]，依次推理，结果全在内存聚合，不写文件。
        """
        self.model.eval()
        self.tokenizer.padding_side = "left"

        results_map_global = collections.defaultdict(list) # Only used on main process
        gold_labels_map_global = {} # Only used on main process

        for i, loader in enumerate(dataloaders):
            view_name = self.cfg.view_names[i]
            tmp_pred=[]
            tmp_label=[]
            
            disable_tqdm = not self.accelerator.is_main_process
            for batch in tqdm(loader, desc=f"{stage} {view_name}", leave=False, disable=disable_tqdm):
                
                inputs = {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"]
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
            

            
            label_map = {"支持": 0, "反对": 1, "中立": 2}
            inv_map = {0: "支持", 1: "反对", 2: "中立"}
            
            local_idxs_tensor = torch.tensor(list(results_map.keys()), device=self.accelerator.device)
            local_preds_list = [label_map.get(results_map[k][0], 2) for k in local_idxs_tensor.tolist()] 
            
            local_preds_tensor = torch.tensor(local_preds_list, device=self.accelerator.device)
            local_gold_list = [label_map.get(gold_labels_map[k], 2) for k in local_idxs_tensor.tolist()]
            local_gold_tensor = torch.tensor(local_gold_list, device=self.accelerator.device)

            all_idxs = self.accelerator.gather(local_idxs_tensor)
            all_preds = self.accelerator.gather(local_preds_tensor)
            all_golds = self.accelerator.gather(local_gold_tensor)
            
            if self.accelerator.is_main_process:
                # Reconstruct full maps on main process
                full_view_preds = []
                full_view_golds = []
                
                # Deduplicate based on idxs (gather can pad)
                seen_idxs = set()
                for idx, p, g in zip(all_idxs.tolist(), all_preds.tolist(), all_golds.tolist()):
                    if idx not in seen_idxs:
                        seen_idxs.add(idx)
                        p_str = inv_map[p]
                        g_str = inv_map[g]
                        
                        full_view_preds.append(p_str)
                        full_view_golds.append(g_str)
                        
                        # Also update global aggregation maps for final voting
                        results_map_global[idx].append(p_str)
                        gold_labels_map_global[idx] = g_str

                tmp_f1=self._compute_char_f1(full_view_preds, full_view_golds)
                print(f"角度：{view_name}   F1: {tmp_f1:.4f}")
                if self.cfg.use_swanlab and SWANLAB_INSTALLED:
                    swanlab.log({f"{view_name}/f1": tmp_f1})

            # Sync before next view
            self.accelerator.wait_for_everyone()

        self.tokenizer.padding_side = "right"

        if self.accelerator.is_main_process:
            final_preds = []
            final_truths = []

            sorted_idxs = sorted(results_map_global.keys())

            for idx in sorted_idxs:
                votes = results_map_global[idx]
                gold = gold_labels_map_global[idx]

                if not votes: votes = ["中立"]
                final_vote = collections.Counter(votes).most_common(1)[0][0]

                final_preds.append(final_vote)
                final_truths.append(gold)

            score = self._compute_char_f1(final_preds, final_truths)
            print(f"Epoch {epoch} {stage.capitalize()} F1 Score: {score:.4f}")

            if self.cfg.use_swanlab and SWANLAB_INSTALLED:
                swanlab.log({f"{stage}/f1": score})

            return score
            
        return 0.0 # Non-main processes return 0

    def _extract_label(self, text):
        match = re.search(r"立场为\s*(支持|反对|中立)", text)
        if match:
            return match.group(1)

        match = re.search(r"(支持|反对|中立)\s*[。！]*\s*$", text)
        if match:
            return match.group(1)

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