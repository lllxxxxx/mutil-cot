import os
import torch
import re
import collections
from tqdm import tqdm
from src.utils import save_results_to_json
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from transformers import get_cosine_schedule_with_warmup
import swanlab
from transformers import logging
from accelerate import Accelerator, DistributedDataParallelKwargs

logging.set_verbosity_error()

SWANLAB_INSTALLED = True


class ManualTrainer:
    def __init__(self, cfg, model, tokenizer, train_dataset, train_collator,
                 val_datasets=None, test_datasets=None, infer_collator=None):
        self.cfg = cfg
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            kwargs_handlers=[ddp_kwargs]
        )
        self.model = model
        self.tokenizer = tokenizer
        self.infer_collator = infer_collator

        # Print distributed info for debugging
        if self.accelerator.is_main_process:
            print(
                f"[INFO] World size: {self.accelerator.num_processes}, Local rank: {self.accelerator.local_process_index}")

        self.optimizer = self._create_optimizer()

        # Prepare model and optimizer
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

        # Create train dataloader and prepare it
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=train_collator,
            num_workers=2,
            pin_memory=True
        )
        self.train_dataloader = self.accelerator.prepare(train_loader)

        # Create val/test dataloaders with DistributedSampler
        # Now accelerator is initialized so we can use proper distributed settings
        self.val_dataloaders = self._create_distributed_dataloaders(val_datasets) if val_datasets else None
        self.test_dataloaders = self._create_distributed_dataloaders(test_datasets) if test_datasets else None

        self.scheduler = self._create_scheduler()
        # Note: Do NOT prepare scheduler with accelerate - it incorrectly divides
        # num_training_steps by num_processes, causing LR to decay too fast

        self.best_metric = -float('inf')

        if self.accelerator.is_main_process:
            os.makedirs(self.cfg.output_dir, exist_ok=True)
            if self.cfg.use_swanlab:
                self._init_tracker()

    def _create_distributed_dataloaders(self, datasets):
        """Create dataloaders with proper distributed sampler for inference."""
        dataloaders = []
        for ds in datasets:
            sampler = DistributedSampler(
                ds,
                num_replicas=self.accelerator.num_processes,
                rank=self.accelerator.process_index,
                shuffle=False
            )
            dl = DataLoader(
                ds,
                batch_size=self.cfg.gen_batch_size,
                shuffle=False,
                sampler=sampler,
                collate_fn=self.infer_collator,
                num_workers=0,
                pin_memory=False
            )
            dataloaders.append(dl)

        return dataloaders

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
        # Calculate total optimization steps
        # In DDP, each GPU processes sharded data, but all GPUs sync gradients together
        # So optimizer steps = len(sharded_dataloader) / gradient_accumulation_steps
        steps_per_epoch = len(self.train_dataloader) // self.cfg.gradient_accumulation_steps
        self.total_steps = steps_per_epoch * self.cfg.num_epochs

        if self.accelerator.is_main_process:
            print(f"[INFO] Scheduler: {len(self.train_dataloader)} batches/epoch, "
                  f"{steps_per_epoch} steps/epoch, {self.total_steps} total steps over {self.cfg.num_epochs} epochs")

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

        eval_step = max(1, int(len(self.train_dataloader) * self.cfg.eval_ratio / self.cfg.gradient_accumulation_steps))

        for epoch in range(self.cfg.num_epochs):
            epoch_total_loss = 0.0
            num_batches = 0

            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.model):
                    inputs = {k: v for k, v in batch.items() if
                              k in ["input_ids", "labels", "attention_mask"] and isinstance(v, torch.Tensor)}

                    outputs = self.model(**inputs)

                    epoch_total_loss += outputs.loss.item()
                    num_batches += 1
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
                                "train/loss_step": current_loss,
                                "train/grad_norm": grad_norm.item(),
                                "train/lr": self.scheduler.get_last_lr()[0],
                                "train/epoch": epoch + (step + 1) / len(self.train_dataloader)
                            }
                            progress_bar.set_postfix(loss=f"{current_loss:.4f}", gn=f"{grad_norm.item():.2f}")
                            if self.cfg.use_swanlab and SWANLAB_INSTALLED:
                                swanlab.log(logs)
                            current_loss = 0.0

                        # Evaluation
                        if global_step > 0 and global_step % eval_step == 0:
                            if self.val_dataloaders:
                                val_score = self.evaluate(self.val_dataloaders, epoch=float(global_step / eval_step),
                                                          stage="val")
                                if self.accelerator.is_main_process and val_score > self.best_metric:
                                    self.best_metric = val_score
                                    print(f" >> New Best Val F1: {self.best_metric:.4f} (Saving...)")
                                    self.save_checkpoint("best_model")
                                    if self.cfg.use_swanlab and SWANLAB_INSTALLED:
                                        swanlab.log({"val/best_f1": self.best_metric})
                                self.model.train()

            self.accelerator.wait_for_everyone()
            self.model.train()

        if self.accelerator.is_main_process:
            progress_bar.close()

        self.accelerator.wait_for_everyone()
        if self.test_dataloaders:
            self.predict_on_test_set()
        if self.accelerator.is_main_process and self.cfg.use_swanlab and SWANLAB_INSTALLED:
            swanlab.finish()

    def save_checkpoint(self, tag):
        """Save checkpoint. Should only be called from main process."""
        path = os.path.join(self.cfg.output_dir, f"checkpoint-{tag}" if isinstance(tag, int) else str(tag))
        os.makedirs(path, exist_ok=True)

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f" >> Checkpoint saved to {path}")

    def predict_on_test_set(self):
        if self.accelerator.is_main_process:
            print("\n*** Loading Best Model for Testing ***")

        # Ensure all processes sync before loading
        self.accelerator.wait_for_everyone()

        # Clear GPU cache before loading adapter to prevent OOM
        torch.cuda.empty_cache()

        path = os.path.join(self.cfg.output_dir, "best_model")

        # All processes need to load the adapter
        try:
            if self.accelerator.is_main_process:
                print(f"  Loading adapter from: {path}")
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.load_adapter(path, adapter_name="default")
            if self.accelerator.is_main_process:
                print("  Adapter loaded successfully!")
        except Exception as e:
            if self.accelerator.is_main_process:
                print(f"  Error loading adapter: {e}")
            return

        # Sync all processes after loading
        self.accelerator.wait_for_everyone()

        self.evaluate(self.test_dataloaders, epoch="TEST", stage="test")

    @torch.inference_mode()
    def evaluate(self, dataloaders, epoch, stage="val"):
        """
        Evaluate on validation/test set. Each process handles its portion of data,
        then results are gathered to the main process for aggregation.
        """
        self.model.eval()
        self.tokenizer.padding_side = "left"

        results_map_global = collections.defaultdict(list)
        gold_labels_map_global = {}
        raw_data_map_global = {}  # Store raw data for JSON output

        label_map = {"支持": 0, "反对": 1, "中立": 2}
        inv_map = {0: "支持", 1: "反对", 2: "中立"}

        for i, loader in enumerate(dataloaders):
            view_name = self.cfg.view_names[i]
            results_map = {}
            gold_labels_map = {}

            # Show progress on main process only
            disable_tqdm = not self.accelerator.is_main_process
            total_batches = len(loader)

            if self.accelerator.is_main_process:
                print(f"\n>>> Evaluating {view_name}: {total_batches} batches")

            for batch in tqdm(loader, desc=f"{stage} {view_name}", leave=True, disable=disable_tqdm, position=0):

                # Move inputs to correct device
                input_ids = batch["input_ids"].to(self.accelerator.device)
                attention_mask = batch["attention_mask"].to(self.accelerator.device)

                unwrapped_model = self.accelerator.unwrap_model(self.model)
                outputs = unwrapped_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=168,
                    do_sample=False,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id
                )

                input_len = input_ids.shape[1]
                decoded_texts = self.tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)

                batch_idxs = batch["original_idx"]
                batch_labels = batch["label_text"]
                batch_targets = batch.get("target", [""] * len(batch_idxs))
                batch_sentences = batch.get("sentences", [[] for _ in range(len(batch_idxs))])
                batch_prompts = batch.get("prompt_text", [""] * len(batch_idxs))
                batch_target_types = batch.get("target_type", [""] * len(batch_idxs))

                for idx, text, gold, target, sentences, prompt, tgt_type in zip(batch_idxs, decoded_texts, batch_labels,
                                                                                batch_targets, batch_sentences,
                                                                                batch_prompts, batch_target_types):
                    if isinstance(idx, torch.Tensor):
                        idx = idx.item()
                    label = self._extract_label(text)
                    results_map[idx] = label
                    gold_labels_map[idx] = gold
                    if idx not in raw_data_map_global:
                        raw_data_map_global[idx] = {
                            "target": target,
                            "target_type": tgt_type,
                            "sentences": sentences,
                            "gold_label": gold,
                            "view_details": {}
                        }

                    raw_data_map_global[idx]["view_details"][view_name] = {
                        "instruction": prompt,
                        "output": text,
                        "pred_label": label
                    }

            # Convert local results to tensors for gathering
            local_idxs_list = list(results_map.keys())
            if len(local_idxs_list) == 0:
                local_idxs_tensor = torch.tensor([], dtype=torch.long, device=self.accelerator.device)
                local_preds_tensor = torch.tensor([], dtype=torch.long, device=self.accelerator.device)
                local_gold_tensor = torch.tensor([], dtype=torch.long, device=self.accelerator.device)
            else:
                local_preds_list = [label_map.get(results_map[k], 2) for k in local_idxs_list]
                local_gold_list = [label_map.get(gold_labels_map[k], 2) for k in local_idxs_list]

                local_idxs_tensor = torch.tensor(local_idxs_list, dtype=torch.long, device=self.accelerator.device)
                local_preds_tensor = torch.tensor(local_preds_list, dtype=torch.long, device=self.accelerator.device)
                local_gold_tensor = torch.tensor(local_gold_list, dtype=torch.long, device=self.accelerator.device)

            # Pad and gather
            all_idxs = self.accelerator.pad_across_processes(local_idxs_tensor, dim=0, pad_index=-1)
            all_preds = self.accelerator.pad_across_processes(local_preds_tensor, dim=0, pad_index=-1)
            all_golds = self.accelerator.pad_across_processes(local_gold_tensor, dim=0, pad_index=-1)

            all_idxs = self.accelerator.gather(all_idxs)
            all_preds = self.accelerator.gather(all_preds)
            all_golds = self.accelerator.gather(all_golds)

            # Gather raw data from all processes
            # Use torch.distributed.all_gather_object as fallback for older accelerate versions
            local_raw_data = []
            for k in local_idxs_list:
                if k in raw_data_map_global:
                    local_raw_data.append((k, raw_data_map_global[k]))

            # Placeholder for gathered results
            all_raw_data_gathered = [None for _ in range(self.accelerator.num_processes)]
            torch.distributed.all_gather_object(all_raw_data_gathered, local_raw_data)

            # Flatten the list of lists
            all_raw_data_list = []
            for sublist in all_raw_data_gathered:
                if sublist:
                    all_raw_data_list.extend(sublist)

            if self.accelerator.is_main_process:
                # Merge gathered raw data into global map
                for idx, data in all_raw_data_list:
                    if idx not in raw_data_map_global:
                        raw_data_map_global[idx] = data
                    else:
                        # Merge view_details if key already exists
                        if "view_details" in data:
                            if "view_details" not in raw_data_map_global[idx]:
                                raw_data_map_global[idx]["view_details"] = {}
                            raw_data_map_global[idx]["view_details"].update(data["view_details"])

                full_view_preds = []
                full_view_golds = []

                seen_idxs = set()
                for idx, p, g in zip(all_idxs.tolist(), all_preds.tolist(), all_golds.tolist()):
                    if idx == -1:
                        continue
                    if idx not in seen_idxs:
                        seen_idxs.add(idx)
                        p_str = inv_map.get(p, "中立")
                        g_str = inv_map.get(g, "中立")

                        full_view_preds.append(p_str)
                        full_view_golds.append(g_str)

                        results_map_global[idx].append(p_str)
                        gold_labels_map_global[idx] = g_str

                tmp_f1 = self._compute_char_f1(full_view_preds, full_view_golds)
                print(f"角度：{view_name}   F1: {tmp_f1:.4f}")
                if self.cfg.use_swanlab and SWANLAB_INSTALLED:
                    swanlab.log({f"{view_name}/f1": tmp_f1})

            self.accelerator.wait_for_everyone()

        self.tokenizer.padding_side = "right"

        if self.accelerator.is_main_process:
            final_preds = []
            final_truths = []

            sorted_idxs = sorted(results_map_global.keys())

            for idx in sorted_idxs:
                votes = results_map_global[idx]
                gold = gold_labels_map_global[idx]

                if not votes:
                    votes = ["中立"]
                final_vote = collections.Counter(votes).most_common(1)[0][0]

                final_preds.append(final_vote)
                final_truths.append(gold)

            print(f"\n--- {stage.capitalize()} Per-Class Metrics ---")
            score = self._compute_char_f1(final_preds, final_truths, print_per_class=True)
            print(f"Epoch {epoch} {stage.capitalize()} Macro F1 Score: {score:.4f}")

            # Save results to JSON (only entries with complete data)
            json_results = []
            for i, idx in enumerate(sorted_idxs):
                if idx not in raw_data_map_global:
                    continue  # Skip indices without raw data (from other processes)
                data = raw_data_map_global[idx].copy()
                data["final_pred"] = final_preds[i]
                json_results.append(data)

            json_path = os.path.join(self.cfg.output_dir, f"{stage}_epoch{epoch}_results.json")
            save_results_to_json(json_results, json_path)

            if self.cfg.use_swanlab and SWANLAB_INSTALLED:
                swanlab.log({f"{stage}/f1": score})

            return score

        return 0.0

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

    def _compute_char_f1(self, preds, truths, print_per_class=False):
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
            if print_per_class:
                print(f"  {label}: P={p:.4f}, R={r:.4f}, F1={f1:.4f}")
        return sum(f1_s) / len(f1_s) if f1_s else 0.0