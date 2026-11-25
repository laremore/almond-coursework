import torch
from src.dataset import get_dataloaders
from src.model import get_base_resnet18

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_resnet18_focal_almond.pth"


def evaluate():
    _, _, test_loader, class_names = get_dataloaders()

    model = get_base_resnet18(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    correct, total = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = outputs.max(1)

            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = 100. * correct / total
    print(f"Test Accuracy: {acc:.2f}%")

    from sklearn.metrics import classification_report
    print(classification_report(all_labels, all_preds, target_names=class_names))


if __name__ == "__main__":
    evaluate()