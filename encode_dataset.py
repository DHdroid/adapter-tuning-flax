from dataclasses import dataclass
import logging
import random
import os
import pickle
import sys
from typing import Tuple

import datasets
from datasets import load_dataset
import hydra
import jax
import numpy as np
from omegaconf import DictConfig, OmegaConf
from flax.training.common_utils import shard
from jax import pmap
from tevatron.arguments import DataArguments
from tevatron.arguments import TevatronTrainingArguments as TrainingArguments
from tevatron.arguments import ModelArguments
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BatchEncoding
from transformers import DataCollatorWithPadding
from transformers import PreTrainedTokenizer

from flax.training.train_state import TrainState
from flax import jax_utils
import optax
from transformers import (AutoConfig, AutoTokenizer, FlaxAutoModel,
                          HfArgumentParser, TensorType)

from src.model.modeling import AdapterBertConfig
from src.model.modeling import FlaxAdapterBertForRetrieval
from src.model.utils import load_adapter_params

logger = logging.getLogger(__name__)


class QueryPreProcessor:
    def __init__(self, tokenizer, query_max_length=32):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length

    def __call__(self, example):
        query_id = example['query_id']
        query = self.tokenizer.encode(example["query"],
                                      add_special_tokens=False,
                                      max_length=self.query_max_length,
                                      truncation=True)
        return {'text_id': query_id, 'text': query}


class CorpusPreProcessor:
    def __init__(self, tokenizer, text_max_length=256, separator=' '):
        self.tokenizer = tokenizer
        self.text_max_length = text_max_length
        self.separator = separator

    def __call__(self, example):
        docid = example['docid']
        text = example['title'] + self.separator + example['text'] if 'title' in example else example['text']
        text = self.tokenizer.encode(text,
                                     add_special_tokens=False,
                                     max_length=self.text_max_length,
                                     truncation=True)
        return {'text_id': docid, 'text': text}


class HFCorpusDataset:
    def __init__(self, tokenizer: PreTrainedTokenizer, dataset_args: DataArguments):
        self.dataset = load_dataset(dataset_args.dataset_name,
                                    dataset_args.dataset_language,
                                    use_auth_token=True)[dataset_args.dataset_split]
        script_prefix = dataset_args.dataset_name
        if script_prefix.endswith('-corpus'):
            script_prefix = script_prefix[:-7]
        self.preprocessor = CorpusPreProcessor
        self.tokenizer = tokenizer
        self.p_max_len = dataset_args.p_max_len
        self.proc_num = dataset_args.dataset_proc_num
        self.separator = getattr(self.tokenizer, dataset_args.passage_field_separator, dataset_args.passage_field_separator)

    def process(self, shard_num=1, shard_idx=0):
        self.dataset = self.dataset.shard(shard_num, shard_idx)
        if self.preprocessor is not None:
            self.dataset = self.dataset.map(
                self.preprocessor(self.tokenizer, self.p_max_len, self.separator),
                batched=False,
                num_proc=self.proc_num,
                remove_columns=self.dataset.column_names,
                desc="Running tokenization",
            )
        return self.dataset


class HFQueryDataset:
    def __init__(self, tokenizer: PreTrainedTokenizer, dataset_args: DataArguments):
        self.dataset = load_dataset(dataset_args.dataset_name,
                                    dataset_args.dataset_language,
                                    use_auth_token=True)[dataset_args.dataset_split]
        self.preprocessor = QueryPreProcessor
        self.tokenizer = tokenizer
        self.q_max_len = dataset_args.q_max_len
        self.proc_num = dataset_args.dataset_proc_num

    def process(self, shard_num=1, shard_idx=0):
        self.dataset = self.dataset.shard(shard_num, shard_idx)
        if self.preprocessor is not None:
            self.dataset = self.dataset.map(
                self.preprocessor(self.tokenizer, self.q_max_len),
                batched=False,
                num_proc=self.proc_num,
                remove_columns=self.dataset.column_names,
                desc="Running tokenization",
            )
        return self.dataset


class EncodeDataset(Dataset):
    input_keys = ['text_id', 'text']

    def __init__(self, dataset: datasets.Dataset, tokenizer: PreTrainedTokenizer, max_len=128):
        self.encode_data = dataset
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.encode_data)

    def __getitem__(self, item) -> Tuple[str, BatchEncoding]:
        text_id, text = (self.encode_data[item][f] for f in self.input_keys)
        encoded_text = self.tok.prepare_for_model(
            text,
            max_length=self.max_len,
            truncation='only_first',
            padding=False,
            return_token_type_ids=False,
        )
        return text_id, encoded_text


@dataclass
class EncodeCollator(DataCollatorWithPadding):
    def __call__(self, features):
        text_ids = [x[0] for x in features]
        text_features = [x[1] for x in features]
        collated_features = super().__call__(text_features)
        # print(collated_features["input_ids"][0])
        # if collated_features["input_ids"].shape[-1] > 30:
        #     mask_m = (np.random.uniform(size=collated_features["input_ids"].shape) < 0.1)
        #     collated_features["input_ids"][mask_m] = self.tokenizer.mask_token_id
        return text_ids, collated_features


@hydra.main(version_base=None, config_path="conf", config_name="encode_config")
def main(args: DictConfig):
    random.seed(42)
    np.random.seed(42)
    model_args = args.model
    dataset_args = args.dataset
    train_args = args.train

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if train_args.local_rank in [-1, 0] else logging.WARN,
    )

    ### Initialized HuggingFace Config
    num_labels = 1
    config = AdapterBertConfig.from_pretrained(
        model_args.model_name,
        num_labels=num_labels
    )
    # model_args.model_name = "facebook/mcontriever-msmarco"
    # config = AutoConfig.from_pretrained(
    #     "facebook/mcontriever-msmarco",
    #     num_labels=num_labels
    # )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name,
        use_fast=False,
    )
    if model_args.adapters:
        config.adapters = OmegaConf.to_object(model_args.adapters)
    else:
        config.adapters = []

    ### Initialize Model w/ Adapter
    model = FlaxAdapterBertForRetrieval.from_pretrained(model_args.model_name, config=config)
    model.params = load_adapter_params(model.params, config)
    # model = FlaxAutoModel.from_pretrained(model_args.model_name, config=config, from_pt=True)

    text_max_length = dataset_args.q_max_len if dataset_args.encode_is_qry else dataset_args.p_max_len
    if dataset_args.encode_is_qry:
        encode_dataset = HFQueryDataset(tokenizer=tokenizer, dataset_args=dataset_args)
    else:
        encode_dataset = HFCorpusDataset(tokenizer=tokenizer, dataset_args=dataset_args)
    encode_dataset = EncodeDataset(encode_dataset.process(dataset_args.encode_num_shard, dataset_args.encode_shard_index),
                                   tokenizer, max_len=text_max_length)

    # prepare padding batch (for last nonfull batch)
    dataset_size = len(encode_dataset)
    padding_prefix = "padding_"
    total_batch_size = len(jax.devices()) * train_args.per_device_eval_batch_size
    features = list(encode_dataset.encode_data.features.keys())
    padding_batch = {features[0]: [], features[1]: []}
    for i in range(total_batch_size - (dataset_size % total_batch_size)):
        padding_batch["text_id"].append(f"{padding_prefix}{i}")
        padding_batch["text"].append([0])
    padding_batch = datasets.Dataset.from_dict(padding_batch)
    encode_dataset.encode_data = datasets.concatenate_datasets([encode_dataset.encode_data, padding_batch])

    encode_loader = DataLoader(
        encode_dataset,
        batch_size=train_args.per_device_eval_batch_size * len(jax.devices()),
        collate_fn=EncodeCollator(
            tokenizer,
            max_length=text_max_length,
            padding='max_length',
            pad_to_multiple_of=16,
            return_tensors=TensorType.NUMPY,
        ),
        shuffle=False,
        drop_last=False,
        num_workers=train_args.dataloader_num_workers,
    )

    # craft a fake state for now to replicate on devices
    adamw = optax.adamw(0.0001)
    state = TrainState.create(apply_fn=model.__call__, params=model.params, tx=adamw)

    def encode_step(batch, state):
        embedding = state.apply_fn(**batch, params=state.params, train=False)[0]
        return embedding[:, 0]

    p_encode_step = pmap(encode_step)
    state = jax_utils.replicate(state)

    encoded = []
    lookup_indices = []

    for (batch_ids, batch) in tqdm(encode_loader):
        lookup_indices.extend(batch_ids)
        batch_embeddings = p_encode_step(shard(batch.data), state)
        encoded.extend(np.concatenate(batch_embeddings, axis=0))

    with open(dataset_args.encoded_save_path, 'wb') as f:
        pickle.dump((encoded[:dataset_size], lookup_indices[:dataset_size]), f)


if __name__ == "__main__":
    main()
