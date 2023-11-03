import json
import pathlib
from typing import Dict, Callable

from datasets import load_dataset, DownloadConfig

TOKEN_HUMAN = '<|human|>'
TOKEN_ASSISTANT = '<|assistant|>'
ALL_TOKENS = [TOKEN_HUMAN, TOKEN_ASSISTANT]

dir_train_data = pathlib.Path(__file__).parent / 'train_data'


def create_download_config() -> DownloadConfig:
    cfg = DownloadConfig()
    cfg.resume_download = True
    cfg.max_retries = 999
    return cfg


def clean_token_surroundings(s: str) -> str:
    for token in ALL_TOKENS:
        s = token.join([line.strip() for line in s.split(token)])
    return s


def convert_row_instruction(row: Dict) -> Dict:
    content: str = row['instruction'] + row['output']
    content = content.replace('Human:', TOKEN_HUMAN)
    content = content.replace('Assistant:', TOKEN_ASSISTANT)
    content = clean_token_surroundings(content)
    return {'text': content}


def save_dataset(dir_train_data: pathlib.Path, name: str, convert: Callable[[Dict], Dict]):
    train_data_file = name.replace('/', '--') + '.jsonl'
    dataset = load_dataset(
        name,
        split='train',
        download_config=create_download_config(),
    )
    with open(dir_train_data / train_data_file, 'w') as f:
        for row in dataset:
            json.dump(convert(row), f, ensure_ascii=False)
            f.write('\r\n')


def main():
    save_dataset(dir_train_data, 'BelleGroup/multiturn_chat_0.8M', convert=convert_row_instruction)


if __name__ == '__main__':
    main()
