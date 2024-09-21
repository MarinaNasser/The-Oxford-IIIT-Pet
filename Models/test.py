import torch
from sklearn.metrics import classification_report
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_model(model, test_loader):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    all_test_labels = []
    all_test_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            all_test_labels.extend(labels.cpu().numpy())
            all_test_preds.extend(preds.cpu().numpy())

    # Convert lists to numpy arrays
    all_test_labels = np.array(all_test_labels)
    all_test_preds = np.array(all_test_preds)

    test_accuracy = (all_test_preds == all_test_labels).mean()

    return loss.item(), test_accuracy, all_test_preds, all_test_labels
