from functools import partial

import datasets


def prepare_dataset(dataset_args,
                    tokenizer):
    ### Initialize Dataset & Dataloader
    train_dataset = \
        datasets.load_dataset(dataset_args.dataset_name,
                              dataset_args.dataset_language,)[dataset_args.dataset_split]
                              #split=f"train[:10%]")#[dataset_args.dataset_split]

    def tokenize_train(example):
        tokenize = partial(tokenizer, return_attention_mask=False, return_token_type_ids=False, padding=False,
                           truncation=True)
        query = example['query']
        pos_psgs = [p['title'] + " " + p['text'] for p in example['positive_passages']]
        neg_psgs = [p['title'] + " " + p['text'] for p in example['negative_passages']]

        example['query_input_ids'] = dict(tokenize(query, max_length=dataset_args.q_max_len))
        example['pos_psgs_input_ids'] = [dict(tokenize(x, max_length=dataset_args.p_max_len)) for x in pos_psgs]
        example['neg_psgs_input_ids'] = [dict(tokenize(x, max_length=dataset_args.p_max_len)) for x in neg_psgs]
        return example

    train_data = train_dataset.map(
        tokenize_train,
        batched=False,
        num_proc=dataset_args.preprocess_num_workers,
        desc="Running tokenizer on train dataset",
        load_from_cache_file=False
    )
    train_data = train_data.filter(
        function=lambda data: len(data["pos_psgs_input_ids"]) >= 1 and \
                              len(data["neg_psgs_input_ids"]) >= dataset_args.train_n_passages-1, num_proc=64
    )
    train_data.cleanup_cache_files()


    class TrainDataset:
        def __init__(self, train_data, group_size, tokenizer):
            self.group_size = group_size
            self.data = train_data
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.data)

        def get_example(self, i, epoch):
            example = self.data[i]
            q = example['query_input_ids']

            pp = example['pos_psgs_input_ids']
            p = pp[0]

            nn = example['neg_psgs_input_ids']
            off = epoch * (self.group_size - 1) % len(nn)
            nn = nn * 2
            nn = nn[off: off + self.group_size - 1]

            return q, [p] + nn

        def get_batch(self, indices, epoch):
            qq, dd = zip(*[self.get_example(i, epoch) for i in map(int, indices)])
            dd = sum(dd, [])
            return dict(tokenizer.pad(qq, max_length=dataset_args.q_max_len, padding='max_length', return_tensors='np')), dict(
                tokenizer.pad(dd, max_length=dataset_args.p_max_len, padding='max_length', return_tensors='np'))

    train_dataset = TrainDataset(train_data, dataset_args.train_n_passages, tokenizer)
    return train_dataset
