# borrowed code from https://github.com/sail-sg/closer-look-LLM-unlearning

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class JBBUnlearningDataset(Dataset):
    def __init__(self, forget_data, retain_data, tokenizer,
                 forget_idk_data=None, retain_idk_data=None, max_length=512):
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.forget_idk_data = forget_idk_data
        self.retain_idk_data = retain_idk_data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.length = max(len(forget_data), len(retain_data))

    def tokenize_with_target(self, goal, target):
        full_text = goal + " " + target
        enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors='pt'
        )
        input_ids = enc['input_ids'].squeeze(0)
        attention_mask = enc['attention_mask'].squeeze(0)

        goal_len = self.tokenizer(
            goal, truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors='pt'
        ).input_ids.shape[1]

        labels = input_ids.clone()
        labels[:goal_len] = -100  # only compute loss on target tokens
        return input_ids, labels, attention_mask

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        f = self.forget_data[idx % len(self.forget_data)]
        r = self.retain_data[idx % len(self.retain_data)]

        forget = self.tokenize_with_target(f['goal'], f['target'])
        retain = self.tokenize_with_target(r['goal'], r['target'])

        # slots 2 and 3 — only populated if idk data provided
        if self.forget_idk_data is not None:
            fi = self.forget_idk_data[idx % len(self.forget_idk_data)]
            forget_idk = self.tokenize_with_target(fi['goal'], fi['target'])
        else: forget_idk = None

        if self.retain_idk_data is not None:
            ri = self.retain_idk_data[idx % len(self.retain_idk_data)]
            retain_idk = self.tokenize_with_target(ri['goal'], ri['target'])
        else: retain_idk = None

        return forget, retain, forget_idk, retain_idk


def _pad_triplet(items):
    """Pad a list of (input_ids, labels, attention_mask) tuples."""
    return (
        pad_sequence([x[0] for x in items], batch_first=True, padding_value=0),
        pad_sequence([x[1] for x in items], batch_first=True, padding_value=-100),
        pad_sequence([x[2] for x in items], batch_first=True, padding_value=0),
    )

def jbb_collator(batch):
    forget      = _pad_triplet([b[0] for b in batch])
    retain      = _pad_triplet([b[1] for b in batch])


    if batch[0][2] is not None: forget_idk = _pad_triplet([b[2] for b in batch])
    else: forget_idk = None

    if batch[0][3] is not None: retain_idk = _pad_triplet([b[3] for b in batch])
    else: retain_idk = None

    return (forget, retain, forget_idk, retain_idk)