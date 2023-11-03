import click
from transformers import BloomTokenizerFast, BloomForCausalLM

BASE_MODEL_NAME = 'Langboat/bloom-800m-zh'


@click.command()
@click.option('--base_model', 'opt_base_model', default=BASE_MODEL_NAME, help='The base model to train.', type=str)
def main(opt_base_model: str):
    tokenizer = BloomTokenizerFast.from_pretrained(opt_base_model)
    model = BloomForCausalLM.from_pretrained(opt_base_model)


if __name__ == '__main__':
    main()
