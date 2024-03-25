import jax
import jax.numpy as jnp


def softmax_cross_entropy_with_integer_labels(
    logits,
    labels
):
    logits_max = jnp.max(logits, axis=-1, keepdims=True)
    logits -= jax.lax.stop_gradient(logits_max)
    label_logits = jnp.take_along_axis(logits, labels[..., None], axis=-1)[..., 0]

    log_normalizers = jnp.log(jnp.sum(jnp.exp(logits), axis=-1))
    ce = log_normalizers - label_logits
    ce = ce.reshape(-1)
    mask = (labels.reshape(-1) != -100)
    return jnp.sum(mask * ce) / jnp.sum(mask)
