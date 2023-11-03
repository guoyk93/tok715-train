from typing import List

import click
from transformers import BloomTokenizerFast, BloomForCausalLM

BASE_MODEL_NAME = 'Langboat/bloom-800m-zh'

TOKEN_HUMAN = '<|human|>'
TOKEN_ASSISTANT = '<|assistant|>'


@click.command()
@click.option('--base_model', 'opt_base_model', default=BASE_MODEL_NAME, help='The base model to train.', type=str)
def main(opt_base_model: str):
    tokenizer = BloomTokenizerFast.from_pretrained(opt_base_model)
    model = BloomForCausalLM.from_pretrained(opt_base_model)

    # add tokens for <|human|> and <|ai|>
    def add_tokens(tokens: List[str]):
        tokenizer.add_tokens(tokens)
        model.resize_token_embeddings(len(tokenizer))
        for token in tokens:
            assert len(tokenizer.encode(token, add_special_tokens=False)) == 1

    add_tokens([TOKEN_HUMAN, TOKEN_ASSISTANT])


if __name__ == '__main__':
    main()
