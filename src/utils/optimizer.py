from functools import partial

from flax import traverse_util
import jax
import jax.numpy as jnp
import optax


def requires_grad(path, adapter_configs):
    for adapter_conf in adapter_configs:
        if f"{adapter_conf['name_prefix']}_adapter" in path and not adapter_conf['freeze']:
            return "grad"
    return "freeze"


def requires_decay(path, adapter_configs):
    for adapter_conf in adapter_configs:
        if f"{adapter_conf['name_prefix']}_adapter" in path and not adapter_conf['freeze']:
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


def get_optimizer(model_params,
                  train_ds_size,
                  adapter_configs,
                  training_args):
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
        training_args.per_device_train_batch_size * jax.device_count(),
        training_args.num_train_epochs,
        training_args.warmup_steps,
        training_args.learning_rate,
    )

    adamw = optax.adamw(
        learning_rate=linear_decay_lr_schedule_fn,
        b1=training_args.adam_beta1,
        b2=training_args.adam_beta2,
        eps=training_args.adam_epsilon,
        weight_decay=training_args.weight_decay,
        mask=partial(create_mask,
                     mask_fn=partial(requires_decay, adapter_configs=adapter_configs)),
    )

    tx = optax.multi_transform({'grad': adamw, 'freeze': zero_grads()},
                                create_mask(model_params, partial(requires_grad, adapter_configs=adapter_configs)))
    return tx
