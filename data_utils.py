import re
import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def tokenize(text):
    return re.findall(r"\b\w+\b|[^\w\s]", text.lower())


def build_vocab(texts, max_vocab_size=20000):
    from collections import Counter

    counter = Counter()

    for text in texts:
        counter.update(tokenize(text))

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}

    for word, _ in counter.most_common(max_vocab_size - 2):
        vocab[word] = len(vocab)

    return vocab


def encode_text(text, vocab, max_length=200):
    tokens = tokenize(text)[:max_length]
    ids = [vocab.get(token, vocab[UNK_TOKEN]) for token in tokens]
    return ids if ids else [vocab[UNK_TOKEN]]


class ReviewDataset(Dataset):
    def __init__(self, examples, vocab, max_length=200):
        self.examples = examples
        self.vocab = vocab
        self.max_len = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        item = self.examples[index]
        ids = encode_text(item["text"], self.vocab, self.max_len)
        label = int(item["label"])

        return torch.tensor(ids), torch.tensor(label)


def collate_batch(batch):
    sequences, labels = zip(*batch)

    lengths = torch.tensor([len(x) for x in sequences])
    max_len = max(lengths).item()

    padded = torch.zeros(
        len(sequences), max_len, dtype=torch.long
    )

    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = seq

    return padded, torch.stack(labels), lengths


def get_data(batch_size=32):
    dataset = load_dataset("stanfordnlp/imdb")

    shuffled = dataset["train"].shuffle(seed=42)

    train_examples = list(shuffled.select(range(4000)))
    val_examples = list(shuffled.select(range(4000, 4500)))
    test_examples = list(
        dataset["test"].shuffle(seed=42).select(range(1000))
    )

    vocab = build_vocab(item["text"] for item in train_examples)

    train_ds = ReviewDataset(train_examples, vocab)
    val_ds = ReviewDataset(val_examples, vocab)
    test_ds = ReviewDataset(test_examples, vocab)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch
    )

    return train_loader, val_loader, test_loader, vocab


def save_vocab(vocab, path="artifacts/vocab.json"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(vocab, file, ensure_ascii=False)


def load_vocab(path="artifacts/vocab.json"):
    with open(path, encoding="utf-8") as file:
        return json.load(file)
