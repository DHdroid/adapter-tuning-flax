def compute_accuracy(preds, labels, ignore=-100):
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics
    labels = labels.reshape(-1)
    preds = preds.reshape(-1)
    mask = (labels != ignore)
    corrected = ((labels == preds) * mask).sum()
    total = mask.sum()
    return corrected / total
