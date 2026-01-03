from typing import List, Dict, Optional, Protocol


class Template(Protocol):
    def format_train(self, item: Dict) -> Dict:
        """Format a single item for training."""
        ...

    def format_predict(self, item: Dict, view_name: Optional[str] = None) -> Dict:
        """Format a single item for prediction."""
        ...


class QwenTemplate:
    def format_train(self, item: Dict) -> Dict:
        user_content = item['instruction'] + "\n" + item.get('input', '')
        instruction = f"{user_content}\n\n###输出：\n"
        response = f"{item['output']}<|endoftext|>"
        return {
            "instruction": instruction,
            "response": response
        }

    def format_predict(self, item: Dict, view_name: Optional[str] = None) -> str:
        if view_name:
            sentences = item['sentences']
            speakers = item['speakers']
            target = item['target']
            dialogue_text = "".join([f"speaker {s}: {t}\n" for s, t in zip(speakers[:-1], sentences[:-1])]) or "无历史会话"

            prompt_template = """您的目标是在给定一段社交媒体中文多轮会话的前提下，判断#当前轮发言#对#指定目标#的立场。可选标签仅包括：#支持#、#反对#、#中立#。

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
            content = prompt_template.format(
                dialogue_text=dialogue_text.strip(),
                current_sentence=f"speaker {speakers[-1]}：{sentences[-1]}",
                target=target,
                view_name=view_name
            )
        else:
            content = item['instruction'] + item.get('input', '')

        return f"{content}\n"


class Llama2Template:
    def format_train(self, item: Dict) -> Dict:
        user_content = item['instruction'] + "\n" + item.get('input', '')
        # Standard Llama 2 Chat format: [INST] {system} {user} [/INST] {assistant}
        # Since we don't have a separate system prompt in the dataset, we'll put everything in user part.
        instruction = f"[INST] {user_content} [/INST]"
        response = f"{item['output']} </s>"  # Llama 2 uses </s> as EOS
        return {
            "instruction": instruction,
            "response": response
        }

    def format_predict(self, item: Dict, view_name: Optional[str] = None) -> str:
        # Reusing the same prompt logic but wrapping in [INST] ... [/INST]
        if view_name:
            sentences = item['sentences']
            speakers = item['speakers']
            target = item['target']
            dialogue_text = "".join([f"speaker {s}: {t}\n" for s, t in zip(speakers[:-1], sentences[:-1])]) or "无历史会话"

            prompt_template = """您的目标是在给定一段社交媒体中文多轮会话的前提下，判断#当前轮发言#对#指定目标#的立场。可选标签仅包括：#支持#、#反对#、#中立#。

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
            content = prompt_template.format(
                dialogue_text=dialogue_text.strip(),
                current_sentence=f"speaker {speakers[-1]}：{sentences[-1]}",
                target=target,
                view_name=view_name
            )
        else:
            content = item['instruction'] + item.get('input', '')

        return f"[INST] {content} [/INST]"


def get_template(name: str):
    if name == "qwen":
        return QwenTemplate()
    elif name == "llama2":
        return Llama2Template()
    else:
        raise ValueError(f"Unknown template: {name}")
