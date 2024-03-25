import os
import random
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

import hydra
from flax import jax_utils
from flax.jax_utils import prefetch_to_device
from flax.training.train_state import TrainState
from flax.training.common_utils import shard
from jax import lax
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, OmegaConf
import optax
from transformers import AutoTokenizer
from tqdm import tqdm
from torch.utils.data import DataLoader, IterableDataset

from src.dataset.mlm_dataset import prepare_dataset
from src.model.utils import load_adapter_params
from src.model.utils import save_adapter_params
from src.utils.optimizer import get_optimizer
from src.dataset.dpr_dataset import prepare_dataset
from src.model.modeling import AdapterBertConfig
from src.model.modeling import FlaxAdapterBertForRetrieval


def _onehot(labels, num_classes: int):
    x = labels[..., None] == jnp.arange(num_classes).reshape((1,) * labels.ndim + (-1,))
    x = lax.select(x, jnp.ones(x.shape), jnp.zeros(x.shape))
    return x.astype(jnp.float32)


def p_contrastive_loss(ss, tt, axis: str = 'device'):
    per_shard_targets = tt.shape[0]
    per_sample_targets = int(tt.shape[0] / ss.shape[0])
    labels = jnp.arange(0, per_shard_targets, per_sample_targets) + per_shard_targets * lax.axis_index(axis)
    
    tt = lax.all_gather(tt, axis).reshape((-1, ss.shape[-1]))
    scores = jnp.dot(ss, jnp.transpose(tt))
    
    label = _onehot(labels, scores.shape[-1])
    return optax.softmax_cross_entropy(scores, label)


def train_step(state, queries, passages, axis='device'):
    def compute_loss(params):
        q_reps = state.apply_fn(**queries, params=params)[0][:, 0, :]
        p_reps = state.apply_fn(**passages, params=params)[0][:, 0, :]
        return jnp.mean(p_contrastive_loss(q_reps, p_reps, axis=axis))

    loss, grad = jax.value_and_grad(compute_loss)(state.params)
    loss, grad = jax.lax.pmean([loss, grad], axis)

    new_state = state.apply_gradients(grads=grad)

    return loss, new_state


@hydra.main(version_base=None, config_path="conf", config_name="dpr_config")
def main(args: DictConfig):
    model_args = args.model
    dataset_args = args.dataset
    train_args = args.train
    random.seed(train_args.seed)
    np.random.seed(train_args.seed)
    ### Initialized HuggingFace Config
    config = AdapterBertConfig.from_pretrained(model_args.model_name)
    if model_args.adapters:
        config.adapters = OmegaConf.to_object(model_args.adapters)
    else:
        config.adapters = []

    ### Initialize Model w/ Adapter
    model = FlaxAdapterBertForRetrieval.from_pretrained(model_args.model_name, config=config)
    # model = FlaxAdapterBertForRetrieval(config)
    model.params = load_adapter_params(model.params, config)

    ### Initialize Tokenizer from Config
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name
    )
    train_dataset = prepare_dataset(dataset_args, tokenizer)
    
    num_epochs = int(train_args.num_train_epochs)
    train_batch_size = int(train_args.per_device_train_batch_size) * jax.device_count()
    train_steps_per_epoch = len(train_dataset) // train_batch_size
    total_train_steps = train_steps_per_epoch * num_epochs

    class IterableTrain(IterableDataset):
        def __init__(self, dataset, batch_idx, epoch):
            super(IterableTrain).__init__()
            self.dataset = dataset
            self.batch_idx = batch_idx
            self.epoch = epoch

        def __iter__(self):
            for idx in self.batch_idx:
                batch = self.dataset.get_batch(idx, self.epoch)
                batch = shard(batch)
                yield batch

    ### Initialize Optimizer
    tx = get_optimizer(model.params, len(train_dataset), config.adapters, train_args)

    ### Build Train State from Model
    state = TrainState.create(apply_fn=model.__call__, params=model.params, tx=tx)
    state = jax_utils.replicate(state)

    ### Training Loop
    p_train_step = jax.pmap(
        train_step,
        axis_name="device"
    )
    rng = jax.random.PRNGKey(train_args.seed)
    for epoch in tqdm(range(train_args.num_train_epochs), desc=f"Epoch ... (1/{train_args.num_train_epochs})", position=0):
        train_losses = []
        steps = tqdm(range(train_steps_per_epoch), desc="Training...", position=1, leave=False)
        rng, input_rng = jax.random.split(rng)

        steps_per_epoch = len(train_dataset) // train_batch_size

        batch_idx = jax.random.permutation(input_rng, len(train_dataset))
        batch_idx = batch_idx[: steps_per_epoch * train_batch_size]
        batch_idx = batch_idx.reshape((steps_per_epoch, train_batch_size)).tolist()

        train_loader = prefetch_to_device(
            iter(DataLoader(
                IterableTrain(train_dataset, batch_idx, epoch),
                num_workers=16, prefetch_factor=256, batch_size=None, collate_fn=lambda v: v)
            ), 2)
        for step in steps:
            # reshape to (B, ...)
            batch = next(train_loader)
            loss, state = p_train_step(state, *batch)
            train_losses.append(loss)
            if step % train_args.train_logging_steps == 0 and step > 0:
                print(f"{step}/{train_steps_per_epoch} | loss = {np.mean(train_losses)}")
                train_losses = []
    
    params = jax_utils.unreplicate(state.params)
    save_adapter_params(params, "task", save_path=f"task_adapter.pickle")


if __name__ == "__main__":
    main()
