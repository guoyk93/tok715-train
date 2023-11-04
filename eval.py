import click
import torch
from transformers import BloomTokenizerFast, BloomForCausalLM, GenerationConfig

from prepare import TOKEN_HUMAN, TOKEN_ASSISTANT
from train import dir_train_output


@click.command()
@click.argument("opt_query", type=str)
def main(opt_query: str):
    tokenizer = BloomTokenizerFast.from_pretrained(
        str(dir_train_output),
    )
    tokenizer.padding_side = 'right'
    assert tokenizer.pad_token_id

    model: BloomForCausalLM = BloomForCausalLM.from_pretrained(
        str(dir_train_output),
        device_map='auto',
        torch_dtype=torch.bfloat16,
    ).eval()

    generation_config = GenerationConfig()
    generation_config.max_new_tokens = 128

    input_text = f'{TOKEN_HUMAN}{opt_query}{TOKEN_ASSISTANT}'

    responses = tokenizer.batch_decode(
        model.generate(
            tokenizer.encode(input_text, return_tensors='pt').cuda(),
            generation_config=generation_config
        ),
    )

    response = responses[0][len(input_text):]

    if response.endswith(tokenizer.eos_token):
        response = response[:-len(tokenizer.eos_token)]

    print(response)


if __name__ == '__main__':
    main()
