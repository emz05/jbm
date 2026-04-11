# borrowed and refactored code from https://github.com/sail-sg/closer-look-LLM-unlearning

import torch
import torch.nn as nn
import torch.nn.functional as F

def _model_device(model):
    if hasattr(model, "module"):
        model = model.module
    return model.get_input_embeddings().weight.device

def _move_triplet_to_device(triplet, device):
    input_ids, labels, attention_mask = triplet
    return (
        input_ids.to(device=device, dtype=torch.long).contiguous().clone(),
        labels.to(device=device, dtype=torch.long).contiguous().clone(),
        attention_mask.to(device=device, dtype=torch.long).contiguous().clone(),
    )

def _validate_input_ids(model, input_ids, name="input_ids"):
    vocab_size = model.get_input_embeddings().weight.shape[0]
    if input_ids.numel():
        min_id = input_ids.min().item()
        max_id = input_ids.max().item()
        if min_id < 0 or max_id >= vocab_size:
            raise ValueError(
                f"{name} must be in [0, {vocab_size}), got [{min_id}, {max_id}]"
            )

def _validate_labels(logits, labels, name="labels"):
    shifted_labels = labels[..., 1:].contiguous()
    valid = shifted_labels[shifted_labels != -100]
    if valid.numel():
        vocab_size = logits.size(-1)
        min_id = valid.min().item()
        max_id = valid.max().item()
        if min_id < 0 or max_id >= vocab_size:
            raise ValueError(
                f"{name} must be in [0, {vocab_size}) or -100, got [{min_id}, {max_id}]"
            )


def get_loss(model, ref_model, inputs, loss_type, beta=0.1):
    # forget_loss
    if 'GA' in loss_type:
        forget_loss = ga_loss(model, inputs)
    elif 'NPO' in loss_type:
        forget_loss = npo_loss(model, ref_model, inputs, beta=beta)
    elif 'DPO' in loss_type:
        forget_loss = dpo_loss(model, ref_model, inputs, beta=beta)
    elif 'ME' in loss_type:
        forget_loss = me_loss(model, inputs)
    elif 'IDK' in loss_type:
        forget_loss = idk_loss(model, inputs)

    # regularization_loss
    if 'GD' in loss_type:
        regularization_loss = gd_loss(model, inputs)
    elif 'KL' in loss_type:
        regularization_loss = kl_loss(model, ref_model, inputs)
    elif 'AP' in loss_type:
        regularization_loss = ap_loss(model, inputs, beta=beta)
    else:
        regularization_loss = 0
    if loss_type == 'LLMU':
        forget_loss = ga_loss(model, inputs)
        regularization_loss = mismatch_loss(model, inputs) + kl_loss(model, ref_model, inputs)

    return forget_loss, regularization_loss


def ga_loss(model, inputs):
    forget_inputs = inputs[0]
    forget_inputs = _move_triplet_to_device(forget_inputs, _model_device(model))
    input_ids, labels, attention_mask = forget_inputs
    _validate_input_ids(model, input_ids)
    outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
    loss = -1 * outputs.loss
    return loss


def npo_loss(model, ref_model, inputs, beta=0.1):
    input_ids, labels, attention_mask = _move_triplet_to_device(
        inputs[0], _model_device(model)
    )
    _validate_input_ids(model, input_ids)

    outputs = model(input_ids, labels=labels,
                    attention_mask=attention_mask)
    _validate_labels(outputs.logits, labels)
    loss_current = get_batch_loss(outputs.logits, labels)

    with torch.no_grad():
        ref_input_ids, ref_labels, ref_attention_mask = _move_triplet_to_device(
            inputs[0], _model_device(ref_model)
        )
        _validate_input_ids(ref_model, ref_input_ids, name="ref_input_ids")
        ref_outputs = ref_model(ref_input_ids, labels=ref_labels,
                                attention_mask=ref_attention_mask)
        _validate_labels(ref_outputs.logits, ref_labels, name="ref_labels")
        loss_ref = get_batch_loss(ref_outputs.logits, ref_labels).to(loss_current.device)

    neg_log_ratios = loss_current - loss_ref
    loss = - F.logsigmoid(beta * neg_log_ratios).mean() * 2 / beta

    return loss


def idk_loss(model, inputs):
    forget_idk_inputs = inputs[2]
    forget_idk_inputs = _move_triplet_to_device(forget_idk_inputs, _model_device(model))
    input_ids, labels, attention_mask = forget_idk_inputs
    _validate_input_ids(model, input_ids)

    outputs = model(input_ids, labels=labels,
                    attention_mask=attention_mask)
    loss = outputs.loss
    return loss


def dpo_loss(model, ref_model, inputs, beta=0.1):
    forget_inputs, forget_idk_inputs = inputs[0], inputs[2]
    forget_input_ids, forget_labels, forget_attention_mask = _move_triplet_to_device(
        forget_inputs, _model_device(model)
    )
    idk_input_ids, idk_labels, idk_attention_mask = _move_triplet_to_device(
        forget_idk_inputs, _model_device(model)
    )
    _validate_input_ids(model, forget_input_ids, name="forget_input_ids")
    _validate_input_ids(model, idk_input_ids, name="idk_input_ids")

    idk_outputs = model(idk_input_ids, labels=idk_labels, attention_mask=idk_attention_mask)
    forget_outputs = model(forget_input_ids, labels=forget_labels, attention_mask=forget_attention_mask)
    _validate_labels(idk_outputs.logits, idk_labels, name="idk_labels")
    _validate_labels(forget_outputs.logits, forget_labels, name="forget_labels")
    idk_loss_current = -1 * get_batch_loss(idk_outputs.logits, idk_labels)
    forget_loss_current = -1 * get_batch_loss(forget_outputs.logits, forget_labels)

    with torch.no_grad():
        ref_forget_input_ids, ref_forget_labels, ref_forget_attention_mask = _move_triplet_to_device(
            forget_inputs, _model_device(ref_model)
        )
        ref_idk_input_ids, ref_idk_labels, ref_idk_attention_mask = _move_triplet_to_device(
            forget_idk_inputs, _model_device(ref_model)
        )
        _validate_input_ids(ref_model, ref_forget_input_ids, name="ref_forget_input_ids")
        _validate_input_ids(ref_model, ref_idk_input_ids, name="ref_idk_input_ids")
        idk_outputs_ref = ref_model(ref_idk_input_ids, labels=ref_idk_labels, attention_mask=ref_idk_attention_mask)
        forget_outputs_ref = ref_model(ref_forget_input_ids, labels=ref_forget_labels, attention_mask=ref_forget_attention_mask)
        _validate_labels(idk_outputs_ref.logits, ref_idk_labels, name="ref_idk_labels")
        _validate_labels(forget_outputs_ref.logits, ref_forget_labels, name="ref_forget_labels")
        idk_loss_ref = -1 * get_batch_loss(idk_outputs_ref.logits, ref_idk_labels).to(idk_loss_current.device)
        forget_loss_ref = -1 * get_batch_loss(forget_outputs_ref.logits, ref_forget_labels).to(forget_loss_current.device)

    pi_logratios = idk_loss_current - forget_loss_current
    ref_logratios = idk_loss_ref - forget_loss_ref
    loss = - F.logsigmoid(beta * (pi_logratios - ref_logratios)).mean() * 2 / beta
    return loss


# Regularization Loss: AP
def get_batch_loss_mean(logits, labels):
    shifted_labels = labels[..., 1:].contiguous()
    logits = logits[..., :-1, :].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    loss = loss_function(logits.transpose(-1, -2), shifted_labels)
    valid = (shifted_labels != -100).float()
    return (loss * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1.0)

def ap_loss(model, inputs, beta=0.1):
    retain_inputs, retain_idk_inputs = inputs[1], inputs[3]
    retain_input_ids, retain_labels, retain_attention_mask = _move_triplet_to_device(
        retain_inputs, _model_device(model)
    )
    retain_idk_input_ids, retain_idk_labels, retain_idk_attention_mask = _move_triplet_to_device(
        retain_idk_inputs, _model_device(model)
    )
    _validate_input_ids(model, retain_input_ids, name="retain_input_ids")
    _validate_input_ids(model, retain_idk_input_ids, name="retain_idk_input_ids")

    outputs = model(retain_input_ids, labels=retain_labels, attention_mask=retain_attention_mask)
    idk_outputs = model(retain_idk_input_ids, labels=retain_idk_labels, attention_mask=retain_idk_attention_mask)
    _validate_labels(outputs.logits, retain_labels, name="retain_labels")
    _validate_labels(idk_outputs.logits, retain_idk_labels, name="retain_idk_labels")

    # loss = get_batch_loss(outputs.logits, retain_labels)
    # loss_idk = get_batch_loss(idk_outputs.logits, retain_idk_labels)
    loss = get_batch_loss_mean(outputs.logits, retain_labels)
    loss_idk = get_batch_loss_mean(idk_outputs.logits, retain_idk_labels)

    neg_log_ratios = loss_idk - loss
    # neg_log_ratios = torch.clamp(neg_log_ratios, min=-10.0, max=10.0)

    loss = - F.logsigmoid(beta * neg_log_ratios).mean() / beta

    return loss

def kl_loss(model, ref_model, inputs):
    input_ids, labels, attention_mask = _move_triplet_to_device(
        inputs[1], _model_device(model)
    )
    _validate_input_ids(model, input_ids)

    outputs = model(input_ids, attention_mask=attention_mask)
    probs = F.log_softmax(outputs.logits[:, :-1, :].float(), dim=-1)

    with torch.no_grad():
        ref_input_ids, _, ref_attention_mask = _move_triplet_to_device(
            inputs[1], _model_device(ref_model)
        )
        outputs_ref = ref_model(ref_input_ids, attention_mask=ref_attention_mask)
    ref_probs = F.log_softmax(outputs_ref.logits[:, :-1, :].float(), dim=-1).to(probs.device)

    per_token_kl = F.kl_div(
        probs, ref_probs,
        reduction='none', log_target=True
    ).sum(dim=-1)

    valid_mask = attention_mask[:, 1:].float()
    denom = valid_mask.sum().clamp_min(1.0)
    loss = (per_token_kl * valid_mask).sum() / denom

    return loss


def mismatch_loss(model, inputs):
    mismatch_inputs = inputs[4]
    mismatch_inputs = _move_triplet_to_device(mismatch_inputs, _model_device(model))
    input_ids, labels, attention_mask = mismatch_inputs
    _validate_input_ids(model, input_ids)

    outputs = model(input_ids, labels=labels,
                    attention_mask=attention_mask)
    _validate_labels(outputs.logits, labels)

    loss = outputs.loss
    return loss


# Regularization Loss: GD
def gd_loss(model, inputs):
    input_ids, labels, attention_mask = _move_triplet_to_device(
        inputs[1], _model_device(model)
    )
    _validate_input_ids(model, input_ids)

    outputs = model(input_ids, labels=labels,
                    attention_mask=attention_mask)
    _validate_labels(outputs.logits, labels)
    loss = outputs.loss
    return loss


def get_batch_loss(logits, labels):
    shifted_labels = labels[..., 1:].contiguous()
    logits = logits[..., :-1, :].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(logits.transpose(-1, -2), shifted_labels).sum(dim=-1)
    return loss


def me_loss(model, inputs):
    input_ids, labels, attention_mask = _move_triplet_to_device(
        inputs[0], _model_device(model)
    )
    _validate_input_ids(model, input_ids)
    outputs = model(input_ids, labels=None, attention_mask=attention_mask)
    loss = get_me_loss(outputs.logits, labels)

    return loss


def get_me_loss(logits, labels):
    num_labels = logits.shape[-1]

    assert logits.shape[:-1] == labels.shape, "Logits and labels must have compatible shapes."

    # Adjust logits and labels to exclude the last token
    labels = labels[:, 1:].clone()  # (bs, seq_len - 1)
    logits = logits[:, :-1, :]  # (bs, seq_len - 1, vocab_size)

    soft_outputs = F.softmax(logits, dim=-1).view(-1, num_labels)  # (bs*seq_len, vocab_size)
    uniform_dist = torch.full_like(soft_outputs, 1.0 / num_labels).to(logits.device)  # (bs*seq_len, vocab_size)

    loss_mask = (labels != -100).view(-1)  # (bs*(seq_len - 1))

    #kl_div = F.kl_div((soft_outputs + 1e-12).log(), uniform_dist, reduction='none').sum(-1)  # (bs*(seq_len - 1))
    kl_div = F.kl_div(uniform_dist.log(), soft_outputs, reduction='none').sum(-1)
    
    masked_kl_div = kl_div * loss_mask  # (bs*(seq_len - 1))
    loss = masked_kl_div.sum() / loss_mask.sum()

    return loss
