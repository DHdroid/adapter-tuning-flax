import os
import random
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

import hydra
from flax import jax_utils
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.dataset.mlm_dataset import prepare_dataset
from src.model.utils import load_adapter_params
from src.model.utils import save_adapter_params
from src.model.modeling import AdapterBertConfig
from src.model.modeling import FlaxAdapterBertForMaskedLM
from src.utils.loss import softmax_cross_entropy_with_integer_labels
from src.utils.metric import compute_accuracy
from src.utils.optimizer import get_optimizer


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


@hydra.main(version_base=None, config_path="conf", config_name="mlm_config")
def main(args: DictConfig):
    random.seed(42)
    np.random.seed(42)
    model_args = args.model
    dataset_args = args.dataset
    train_args = args.train
    ### Initialized HuggingFace Config
    config = AdapterBertConfig.from_pretrained(model_args.model_name)
    if model_args.adapters:
        config.adapters = OmegaConf.to_object(model_args.adapters)
    else:
        config.adapters = []

    ### Initialize Model w/ Adapter
    model = FlaxAdapterBertForMaskedLM.from_pretrained(model_args.model_name, config=config)
    model.params = load_adapter_params(model.params, config)

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
    tx = get_optimizer(model.params, train_ds_size, config.adapters, train_args)


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
            batch = next(train_loader)
            # reshape to (B, ...)
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
    
    params = jax_utils.unreplicate(state.params)
    save_adapter_params(params, "language", save_path=f"{dataset_args.dataset_config_name}.pkl")


if __name__ == "__main__":
    main()
