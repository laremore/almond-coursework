import torch
import torch.nn as nn
import torch.optim as optim
from clearml import Task
from src.dataset import get_dataloaders
from src.model import ResNet18CBAM, get_base_resnet18

EXPERIMENT_NAME = "resnet18_focal_almond"
USE_CBAM = False
EPOCHS = 20
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=1.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

def train():
    task = Task.init(
        project_name="Almond Classification Coursework",
        task_name=EXPERIMENT_NAME
    )

    task.connect({
        "model": "ResNet18" + ("+CBAM" if USE_CBAM else ""),
        "epochs": EPOCHS,
        "lr": LR,
        "batch_size": 32,
        "optimizer": "Adam"
    })

    train_loader, val_loader, _, class_names = get_dataloaders()
    print("Classes:", class_names)

    if USE_CBAM:
        model = ResNet18CBAM(num_classes=len(class_names), pretrained=True)
    else:
        model = get_base_resnet18(num_classes=len(class_names), pretrained=True)
    model = model.to(DEVICE)

    criterion = FocalLoss(gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = outputs.max(1)
            train_total += labels.size(0)
            train_correct += preds.eq(labels).sum().item()

        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, preds = outputs.max(1)
                val_total += labels.size(0)
                val_correct += preds.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        task.get_logger().report_scalar("Loss", "train", avg_train_loss, epoch)
        task.get_logger().report_scalar("Loss", "val", avg_val_loss, epoch)
        task.get_logger().report_scalar("Accuracy", "train", train_acc, epoch)
        task.get_logger().report_scalar("Accuracy", "val", val_acc, epoch)

        print(f"Epoch {epoch + 1}/{EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"best_{EXPERIMENT_NAME}.pth")
            print(f"  -> Saved best model (val_acc={val_acc:.2f}%)")

    print(f"Training done. Best val_acc: {best_acc:.2f}%")
    task.close()


if __name__ == "__main__":
    train()