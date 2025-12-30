import json
import os
import sys
import asyncio
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
from transformers import HfArgumentParser

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import TrainConfig


def format_dialogue(sentences, speakers):
    text = ""
    for sent, spk in zip(sentences, speakers):
        text += f"User {spk}: {sent}\n"
    return text.strip()


def construct_sample(dialogue, current, target, view_name, cot, label_text, current_speaker):

    instruction = f"""您的目标是在给定一段社交媒体中文多轮会话的前提下，判断#当前轮发言#对#指定目标#的立场。可选标签仅包括：#支持#、#反对#、#中立#。

请从#{view_name}#的角度进行立场分析。"""

    input_text = f"""###输入：
- 会话历史：
{dialogue}

- 当前轮发言：
{current}

- 指定目标：
{target}"""

    # 3. output
    output = f"【立场倾向性分析】{cot}【通过对上述分析综合研判】：用户{current_speaker}对{target}的立场为{label_text}"

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output
    }


async def process_single_item(client, sem, item, template, view_name, label_map, model_name):
    async with sem:
        dialogue = format_dialogue(item['sentences'][:-1], item['speakers'][:-1]) or "无历史对话"
        current = item['sentences'][-1]
        target = item['target']
        label = label_map.get(item['label'], "中立")

        full_prompt = template.format(target=target, dialogue_text=dialogue, current_sentence=current)

        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.7
            )
            cot = response.choices[0].message.content
            return construct_sample(dialogue, current, target, view_name, cot, label, item['speakers'][-1])

        except Exception as e:
            return None


async def main():
    parser = HfArgumentParser((TrainConfig,))
    cfg, = parser.parse_args_into_dataclasses()

    client = AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
    sem = asyncio.Semaphore(cfg.api_concurrency)

    with open(cfg.raw_data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    prompt_templates = []
    for p_file in cfg.prompt_files:
        with open(p_file, 'r', encoding='utf-8') as f:
            prompt_templates.append(f.read())

    label_map = {0: "支持", 1: "反对", 2: "中立"}

    for view_name, output_path, template in zip(cfg.view_names, cfg.generated_data_paths, prompt_templates):
        print(f"Generating View: {view_name} -> {output_path}")
        tasks = []
        for item in raw_data:
            tasks.append(process_single_item(client, sem, item, template, view_name, label_map, cfg.api_model))

        results = await tqdm.gather(*tasks)
        processed_data = [item for item in results if item is not None]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    asyncio.run(main())