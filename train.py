import os
import torch
import torch.nn as nn

from data_utils import get_data, save_vocab
from model import SentimentModel


def evaluate_loss(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for ids, labels, lengths in loader:
            ids = ids.to(device)
            labels = labels.to(device)

            logits, _ = model(ids, lengths)
            loss = criterion(logits, labels)

            total_loss += loss.item() * len(labels)
            total_correct += (
                logits.argmax(dim=1) == labels
            ).sum().item()

            total_examples += len(labels)

    avg_loss = total_loss / total_examples
    accuracy = total_correct / total_examples

    return avg_loss, accuracy


def main():
    os.makedirs("artifacts", exist_ok=True)
    torch.manual_seed(42)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Device:", device)

    train_loader, val_loader, _, vocab = get_data()

    model = SentimentModel(vocab_size=len(vocab)).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    epochs = 5

    for epoch in range(epochs):
        model.train()

        total_train_loss = 0.0
        total_examples = 0

        for ids, labels, lengths in train_loader:
            ids = ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits, _ = model(ids, lengths)
            loss = criterion(logits, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            total_train_loss += loss.item() * len(labels)
            total_examples += len(labels)

        train_loss = total_train_loss / total_examples

        val_loss, val_accuracy = evaluate_loss(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                "artifacts/model.pt"
            )

            save_vocab(vocab)
            print("Best model and vocabulary saved.")

    print("Training complete.")


if __name__ == "__main__":
    main()
