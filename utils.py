import jax
import jax.numpy as jnp


def compute_accuracy(preds, labels):
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics
    labels = labels.reshape(-1)
    preds = preds.reshape(-1)
    mask = (labels != -100)
    corrected = ((labels == preds) * mask).sum()
    total = mask.sum()
    return corrected / total


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
