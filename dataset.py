from functools import partial
from itertools import chain

from datasets import load_dataset
import datasets


def group_texts(examples,
                max_seq_length: int):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < max_seq_length  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // max_seq_length) * max_seq_length
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
        for k, t in concatenated_examples.items()
    }
    return result


def tokenize_function(examples, tokenizer, text_column_name):
    return tokenizer(examples[text_column_name],
                     return_special_tokens_mask=True,
                     return_tensors="np")


def get_raw_datasets(dataset_name: str,
                    dataset_config_name: str):
    raw_datasets = load_dataset(
        dataset_name,
        dataset_config_name
    )
    if "validation" not in raw_datasets.keys():
        raw_datasets["validation"] = load_dataset(
            dataset_name,
            dataset_config_name,
            split=f"train[:20%]",
        )
        raw_datasets["train"] = load_dataset(
            dataset_name,
            dataset_config_name,
            split=f"train[20%:]",
        )
    return raw_datasets


def prepare_dataset(tokenizer,
                    args):
    max_seq_length = tokenizer.model_max_length
    raw_datasets = get_raw_datasets(args.dataset_name,
                                    args.dataset_config_name)
    tokenized_datasets = raw_datasets.map(
        partial(tokenize_function,
                tokenizer=tokenizer,
                text_column_name=args.text_column_name),
        batched=True,
        num_proc=args.preprocess_num_workers,
        remove_columns=list(raw_datasets["train"].features),
    )
    tokenized_datasets = tokenized_datasets.map(
        partial(group_texts,
                max_seq_length=max_seq_length),
        batched=True,
        num_proc=args.preprocess_num_workers,
    )
    return tokenized_datasets
