import json
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from transformers import DataCollatorForSeq2Seq


def prepare_4d_attention_mask(attention_mask_with_indices: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    bsz, seq_len = attention_mask_with_indices.size()
    min_dtype = torch.finfo(dtype).min
    non_padding_mask = (attention_mask_with_indices != 0).unsqueeze(1).unsqueeze(2)
    tril_mask = torch.tril(torch.ones((seq_len, seq_len), device=attention_mask_with_indices.device, dtype=torch.bool))
    tril_mask = tril_mask.unsqueeze(0).unsqueeze(0)
    attention_mask_4d = tril_mask & non_padding_mask
    attention_mask_4d = torch.where(attention_mask_4d, torch.tensor(0, dtype=dtype), min_dtype)
    return attention_mask_4d


class QwenSFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length,seed):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        if isinstance(data_path, str): data_path = [data_path]
        rng = random.Random(seed)
        d1 = json.load(open(data_path[0], "r", encoding="utf-8"))
        n = len(d1)
        k, r = divmod(n, 3)
        parts = [k + (i < r) for i in range(3)]

        for i,path in enumerate(data_path):
            data=json.loads(open(path, "r", encoding="utf-8").read())
            self.data.extend(rng.sample(data, parts[0]))
        rng.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        user_content = item['instruction'] + "\n"+item.get('input', '')

        instruction = f"{user_content}\n\n###输出：\n"
        response = f"{item['output']}<|endoftext|>"

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


class QwenPredictDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length, view_name=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.view_name = view_name  # 接收视角名称

        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.prompt_template = """您的目标是在给定一段社交媒体中文多轮会话的前提下，判断#当前轮发言#对#指定目标#的立场。可选标签仅包括：#支持#、#反对#、#中立#。

请从#{view_name}#的角度进行立场分析。
###输入：
- 历史会话：
{dialogue_text}


- 当前轮发言：
{current_sentence}


- 指定目标：
{target}

###输出：
"""

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        if self.view_name:
            sentences = item['sentences']
            speakers = item['speakers']
            dialogue_text = "".join([f"speaker {s}: {t}\n" for s, t in zip(speakers[:-1], sentences[:-1])]) or "无历史会话"

            content = self.prompt_template.format(
                dialogue_text=dialogue_text.strip(),
                current_sentence=f"speaker {speakers[-1]}：{sentences[-1]}",
                target=item['target'],
                view_name=self.view_name
            )
            label_text = {0: "支持", 1: "反对", 2: "中立"}.get(item['label'], "中立")
        else:
            content = item['instruction'] + item.get('input', '')
            label_text = item.get('output', '')

        prompt_text = f"{content}\n"

        input_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor([1] * len(input_ids), dtype=torch.long),
            "prompt_text": prompt_text,
            "label_text": label_text,
            "original_idx": idx
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

        batch = super().__call__(features_for_model)

        if self.use_4d_mask and not self.for_inference and not self.use_flash_attn and "attention_mask" in batch:
            dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
            batch["attention_mask"] = prepare_4d_attention_mask(batch["attention_mask"], dtype)

        if raw_prompts: batch["prompt_text"] = raw_prompts
        if raw_labels: batch["label_text"] = raw_labels
        if original_idxs: batch["original_idx"] = original_idxs

        return batch