import copy
import pathlib
from typing import List, Dict, Sequence

import click
import torch
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq, \
    BloomTokenizerFast, BloomForCausalLM

from prepare import ALL_TOKENS, dir_train_data

dir_train_output = pathlib.Path(__file__).parent / 'train_output'

IGNORE_INDEX = -100


@click.command()
@click.option(
    '--base_model', 'opt_base_model',
    default='Langboat/bloom-800m-zh',
    help='The base model to train.',
    type=str,
)
def main(opt_base_model: str):
    tokenizer = BloomTokenizerFast.from_pretrained(
        opt_base_model,
    )
    tokenizer.padding_side = 'right'
    assert tokenizer.pad_token_id

    model = BloomForCausalLM.from_pretrained(
        opt_base_model,
        device_map='cuda',
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
    def tokenize_train_dataset_data(strings: Sequence[str]) -> Dict:
        tokenized_list = [
            tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
            )
            for text in strings
        ]
        input_ids = labels = [
            tokenized.input_ids[0]
            for tokenized in tokenized_list
        ]
        ne_pad_token_id = IGNORE_INDEX if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        input_ids_lens = labels_lens = [
            tokenized.input_ids.ne(ne_pad_token_id).sum().item() for tokenized in tokenized_list
        ]
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

    def preprocess_train_dataset_data(
            sources: Sequence[str],
            targets: Sequence[str],
    ) -> Dict:
        examples = [s + t for s, t in zip(sources, targets)]
        examples_tokenized, sources_tokenized = (
            tokenize_train_dataset_data(examples),
            tokenize_train_dataset_data(sources)
        )
        examples_input_ids = examples_tokenized["input_ids"]
        examples_labels = copy.deepcopy(examples_input_ids)
        for label, source_len in zip(examples_labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = IGNORE_INDEX
        return dict(input_ids=examples_input_ids, labels=examples_labels)

    data_files = list(filter(lambda x: x.endswith('.jsonl'), [
        str(f) for f in dir_train_data.iterdir()
    ]))
    train_dataset = load_dataset('json', data_files=data_files, split='train')

    def map_train_dataset_batched_rows(batched_rows: Dict) -> any:
        input_output = preprocess_train_dataset_data(
            sources=batched_rows['instruction'],
            targets=[output + tokenizer.eos_token for output in batched_rows['output']],
        )
        return dict(
            input_ids=input_output['input_ids'],
            labels=input_output['labels'],
        )

    train_dataset = train_dataset.map(
        map_train_dataset_batched_rows,
        batched=True,
        num_proc=8,
    )

    # train args
    args = TrainingArguments(
        output_dir=str(dir_train_output),
        num_train_epochs=3,
        learning_rate=2e-5,
        bf16=False,
        tf32=False,
        per_device_train_batch_size=10,
        per_device_eval_batch_size=6,
        gradient_accumulation_steps=8,
        evaluation_strategy='no',
        save_strategy='steps',
        save_steps=2000,
        save_total_limit=10,
        logging_steps=100,
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
            label_pad_token_id=IGNORE_INDEX,
        )
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model()


if __name__ == '__main__':
    main()
