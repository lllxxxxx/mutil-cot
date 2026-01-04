import json
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from transformers import DataCollatorForSeq2Seq
from src.templates import get_template


def prepare_4d_attention_mask(attention_mask_with_indices: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    bsz, seq_len = attention_mask_with_indices.size()
    min_dtype = torch.finfo(dtype).min
    non_padding_mask = (attention_mask_with_indices != 0).unsqueeze(1).unsqueeze(2)
    tril_mask = torch.tril(torch.ones((seq_len, seq_len), device=attention_mask_with_indices.device, dtype=torch.bool))
    tril_mask = tril_mask.unsqueeze(0).unsqueeze(0)
    attention_mask_4d = tril_mask & non_padding_mask
    attention_mask_4d = torch.where(attention_mask_4d, torch.tensor(0, dtype=dtype), min_dtype)
    return attention_mask_4d


class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length, seed, template_name="qwen"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        if isinstance(data_path, str): data_path = [data_path]
        rng = random.Random(seed)
        for i, path in enumerate(data_path):
            data = json.load(open(path, "r", encoding="utf-8"))
            self.data.extend(data)
        rng.shuffle(self.data)
        self.template = get_template(template_name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        formatted = self.template.format_train(item)
        instruction = formatted["instruction"]
        response = formatted["response"]

        enc_instr = self.tokenizer.encode(instruction, add_special_tokens=False)
        enc_res = self.tokenizer.encode(response, add_special_tokens=False)

        input_ids = enc_instr + enc_res
        labels = [-100] * len(enc_instr) + enc_res

        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }


class PredictDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length, view_name=None, template_name="qwen"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.view_name = view_name  # 接收视角名称
        self.template = get_template(template_name)

        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        prompt_text = self.template.format_predict(item, self.view_name)

        if self.view_name:
            target = item['target']
            sentences = item['sentences']
            label_text = {0: "支持", 1: "反对", 2: "中立"}.get(item['label'], "中立")
        else:
            label_text = item.get('output', '')
            target = item.get('target', '')
            sentences = item.get('sentences', [])

        input_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor([1] * len(input_ids), dtype=torch.long),
            "prompt_text": prompt_text,
            "label_text": label_text,
            "original_idx": idx,
            "target": target,
            "sentences": sentences,
            "view_name": self.view_name or "",
            "target_type":item["target_type"]
        }


class ManualCollator(DataCollatorForSeq2Seq):
    def __init__(self, tokenizer, use_flash_attn=False, for_inference=False, use_4d_mask=True):
        super().__init__(tokenizer, padding=True)
        self.use_flash_attn = use_flash_attn
        self.for_inference = for_inference
        self.use_4d_mask = use_4d_mask

    @staticmethod
    def _to_list(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if torch.is_tensor(x):
            return x.tolist()
        return x

    def __call__(self, features):
        features_for_model = []
        raw_prompts = []
        raw_labels = []
        original_idxs = []
        targets = []
        sentences_list = []
        view_names = []
        target_type=[]

        for f in features:
            clean_dict = {}
            if "input_ids" in f:
                clean_dict["input_ids"] = self._to_list(f["input_ids"])
            if "labels" in f:
                clean_dict["labels"] = self._to_list(f["labels"])
            if "attention_mask" in f:
                clean_dict["attention_mask"] = self._to_list(f["attention_mask"])

            features_for_model.append(clean_dict)

            if "prompt_text" in f: raw_prompts.append(f["prompt_text"])
            if "label_text" in f: raw_labels.append(f["label_text"])
            if "original_idx" in f: original_idxs.append(f["original_idx"])
            if "target" in f: targets.append(f["target"])
            if "sentences" in f: sentences_list.append(f["sentences"])
            if "view_name" in f: view_names.append(f["view_name"])
            if "target_type" in f: target_type.append(f["target_type"])   

        batch = super().__call__(features_for_model)

        if self.use_4d_mask and not self.for_inference and not self.use_flash_attn and "attention_mask" in batch:
            dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
            batch["attention_mask"] = prepare_4d_attention_mask(batch["attention_mask"], dtype)

        if raw_prompts: batch["prompt_text"] = raw_prompts
        if raw_labels: batch["label_text"] = raw_labels
        if original_idxs: batch["original_idx"] = original_idxs
        if targets: batch["target"] = targets
        if sentences_list: batch["sentences"] = sentences_list
        if view_names: batch["view_name"] = view_names
        if target_type: batch["target_type"] = target_type

        return batch