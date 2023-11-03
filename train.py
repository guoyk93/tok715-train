import pathlib
from typing import List, Dict

import click
import torch
from datasets import load_dataset
from transformers import BloomTokenizerFast, BloomForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq

from prepare import ALL_TOKENS, dir_train_data


@click.command()
@click.option(
    '--base_model', 'opt_base_model',
    default='Langboat/bloom-800m-zh',
    help='The base model to train.',
    type=str,
)
def main(opt_base_model: str):
    dir_train_output = pathlib.Path(__file__).parent / 'train_output'

    tokenizer = BloomTokenizerFast.from_pretrained(
        opt_base_model,
        use_fast=False,
    )

    model = BloomForCausalLM.from_pretrained(
        opt_base_model,
        device_map='auto',
        torch_dtype=torch.bfloat16,
    ).eval()

    # add extra tokens
    def add_tokens(tokens: List[str]):
        tokenizer.add_tokens(tokens)
        model.resize_token_embeddings(len(tokenizer))
        for token in tokens:
            assert len(tokenizer.encode(token, add_special_tokens=False)) == 1

    add_tokens(ALL_TOKENS)

    # load train data
    data_files = list(filter(lambda x: x.endswith('.jsonl'), [
        str(f) for f in dir_train_data.iterdir()
    ]))
    train_dataset = load_dataset('json', data_files=data_files, split='train')

    def map_tokens(row: Dict) -> any:
        return tokenizer(row['text'])

    train_dataset = train_dataset.map(map_tokens)

    # train args
    args = TrainingArguments(
        output_dir=str(dir_train_output),
        num_train_epochs=3,
        bf16=False,
        per_device_train_batch_size=1,
        per_gpu_eval_batch_size=1,
        gradient_accumulation_steps=8,
        evaluation_strategy='no',
        save_strategy='steps',
        save_steps=20,
        save_total_limit=10,
        learning_rate=2e-5,
        logging_steps=10,
        tf32=False,
    )

    # trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=tokenizer.pad_token_id,
        )
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model()


if __name__ == '__main__':
    main()
