import argparse
from functools import partial
import yaml
import os
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

import hydra
from flax import jax_utils
from flax import traverse_util
from flax.jax_utils import prefetch_to_device
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np

from omegaconf import DictConfig, OmegaConf
import optax
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import prepare_dataset
from modeling import AdapterBertConfig
from modeling import FlaxAdapterBertForMaskedLM
from utils import softmax_cross_entropy_with_integer_labels
from utils import compute_accuracy

def requires_grad(path, config):
    for i in range(config.num_adapters):
        if f"bert_adapter_{i}" in path:
            return "grad"
    return "freeze"


def requires_decay(path, config):
    for i in range(config.num_adapters):
        if f"bert_adapter_{i}" in path:
            return True
    return False


def create_mask(params, mask_fn):
    flat_params = traverse_util.flatten_dict(params)

    flat_mask = {path: mask_fn(path) for path in flat_params}
    return traverse_util.unflatten_dict(flat_mask)


def zero_grads():
    # from https://github.com/deepmind/optax/issues/159#issuecomment-896459491
    def init_fn(_): 
        return ()
    def update_fn(updates, state, params=None):
        return jax.tree_map(jnp.zeros_like, updates), ()
    return optax.GradientTransformation(init_fn, update_fn)


def get_optimizer(train_ds_size,
                  config,
                  args):
    def create_learning_rate_fn(
            train_ds_size: int, train_batch_size: int, num_train_epochs: int, num_warmup_steps: int,
            learning_rate: float
        ):
        """Returns a linear warmup, linear_decay learning rate function."""
        steps_per_epoch = train_ds_size // train_batch_size
        num_train_steps = steps_per_epoch * num_train_epochs
        warmup_fn = optax.linear_schedule(init_value=0.0, end_value=learning_rate, transition_steps=num_warmup_steps)
        decay_fn = optax.linear_schedule(
            init_value=learning_rate, end_value=0, transition_steps=num_train_steps - num_warmup_steps
        )
        schedule_fn = optax.join_schedules(schedules=[warmup_fn, decay_fn], boundaries=[num_warmup_steps])
        return schedule_fn

    linear_decay_lr_schedule_fn = create_learning_rate_fn(
        train_ds_size,
        args.per_device_train_batch_size * jax.device_count(),
        args.num_train_epochs,
        args.warmup_steps,
        args.learning_rate,
    )

    adamw = optax.adamw(
        learning_rate=linear_decay_lr_schedule_fn,
        b1=args.adam_beta1,
        b2=args.adam_beta2,
        eps=args.adam_epsilon,
        weight_decay=args.weight_decay,
        mask=partial(create_mask,
                     mask_fn=partial(requires_decay, config=config)),
    )

    return adamw


def train_step(state, batch, axis="device"):
    labels = batch.pop("labels")
    def compute_loss(params, labels):
        output = state.apply_fn(**batch, params=params)
        loss = softmax_cross_entropy_with_integer_labels(output.logits, labels)
        accuracy = compute_accuracy(jnp.argmax(output.logits, -1), labels)
        return loss, accuracy
    (loss, acc), grad = jax.value_and_grad(compute_loss, has_aux=True)(state.params, labels)
    loss, acc, grad = jax.lax.pmean([loss, acc, grad], axis)

    new_state = state.apply_gradients(grads=grad)
    return loss, acc, new_state


def validation_step(state, batch, axis="device"):
    labels = batch.pop("labels")
    def compute_loss(params, labels):
        output = state.apply_fn(**batch, params=params)
        loss = softmax_cross_entropy_with_integer_labels(output.logits, labels)
        accuracy = compute_accuracy(jnp.argmax(output.logits, -1), labels)
        return loss, accuracy
    loss, accuracy = compute_loss(state.params, labels)
    loss, accuracy = jax.lax.pmean([loss, accuracy], axis)
    return loss, accuracy


def shard(batch):
    sharded = dict()
    for key in batch.keys():
        sharded[key] = batch[key].reshape((jax.device_count(), -1) + batch[key].shape[1:])
    return sharded


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args: DictConfig):
    model_args = args.model
    dataset_args = args.dataset
    train_args = args.train
    ### Initialized HuggingFace Config
    config = AdapterBertConfig.from_pretrained(model_args.model_name)
    config.num_adapters = model_args.num_adapters
    config.adapter_reduce_factor = model_args.adapter_reduce_factor

    ### Initialize Model w/ Adapter
    model = FlaxAdapterBertForMaskedLM.from_pretrained(model_args.model_name, config=config)

    ### Initialize Tokenizer from Config
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name
    )

    ### Initialize Dataset & Dataloader
    dataset = prepare_dataset(tokenizer, dataset_args)
    train_ds_size = len(dataset["train"])
    validation_ds_size = len(dataset["validation"])

    total_train_batch_size = train_args.per_device_train_batch_size * jax.device_count()
    total_validation_batch_size = train_args.per_device_validation_batch_size * jax.device_count()

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=train_args.mlm_probability,
        return_tensors="np",
    )
    train_dataloader = DataLoader(dataset["train"],
                                  batch_size=total_train_batch_size,
                                  num_workers=dataset_args.num_workers,
                                  collate_fn=collator)
    validation_dataloader = DataLoader(dataset["validation"],
                                       batch_size=total_validation_batch_size,
                                       num_workers=dataset_args.num_workers,
                                       collate_fn=collator)
    train_steps_per_epoch = train_ds_size // total_train_batch_size
    validation_steps_per_epoch = validation_ds_size // total_validation_batch_size

    ### Initialize Optimizer
    opt = get_optimizer(train_ds_size, config, train_args)
    tx = optax.multi_transform({'grad': opt, 'freeze': zero_grads()},
                                create_mask(model.params, partial(requires_grad, config=config)))

    ### Build Train State from Model
    state = TrainState.create(apply_fn=model.__call__, params=model.params, tx=tx)
    state = jax_utils.replicate(state)

    ### Training Loop
    p_train_step = jax.pmap(
        train_step,
        axis_name="device"
    )
    p_validation_step = jax.pmap(
        validation_step,
        axis_name="device"
    )
    for epoch in tqdm(range(train_args.num_train_epochs), desc=f"Epoch ... (1/{train_args.num_train_epochs})", position=0):
        train_losses = []
        steps = tqdm(range(train_steps_per_epoch), desc="Training...", position=1, leave=False)
        train_loader = iter(train_dataloader)
        for step in steps:
            # reshape to (B, ...)
            batch = next(train_loader)
            batch = shard(batch)
            loss, acc, state = p_train_step(state, batch)
            train_losses.append(loss)
            if step % train_args.train_logging_steps == 0 and step > 0:
                print(f"{step}/{train_steps_per_epoch} | loss = {np.mean(train_losses)}")
                train_losses = []

        validation_losses = []
        validation_accs = []
        steps = tqdm(range(validation_steps_per_epoch), desc="Validating...", position=1, leave=False)
        validation_loader = iter(validation_dataloader)
        for step in steps:
            batch = next(validation_loader)
            # reshape to (B, ...)
            batch = shard(batch)
            loss, acc = p_validation_step(state, batch)
            validation_losses.append(loss)
            validation_accs.append(acc)
        print(f"Epoch {epoch} | validation loss = {np.mean(validation_losses)}, accuracy = {np.mean(validation_accs)}")


if __name__ == "__main__":
    main()
