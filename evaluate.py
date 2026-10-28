import torch
from sklearn.metrics import(accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix)

from data_utils import get_data
from model import SentimentModel

def main():
    _, _, test_loader, vocab = get_data()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    model = SentimentModel(vocab_size=len(vocab)).to(device)
    model.load_state_dict(
        torch.load(
            "artifacts/model.pt",
            map_location = device,
            weights_only = True
        )
    )
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for ids, labels, lengths in test_loader:
            ids = ids.to(device)
            logits, _ = model(ids, lengths)

            predictions = logits.argmax(dim = 1).cpu().tolist()
            all_predictions.extend(predictions)
            all_labels.extend(labels.tolist())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average="binary",
            zero_division=0
        )
    )
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\nClassification report:")
    print(classification_report(
        all_labels,
        all_predictions,
        target_names=["Negative", "Positive"],
        zero_division=0
    ))

    print("Confusion matrix:")
    print(confusion_matrix(all_labels, all_predictions))


if __name__ == "__main__":
    main()