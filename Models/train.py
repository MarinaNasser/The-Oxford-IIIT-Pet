import torch

# Define device (make sure this is included)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, criterion, optimizer, train_loader, num_epochs=10):

    train_losses = []
    train_accuracies = []

    best_value = 0
    for epoch in range(num_epochs):

        # Training Phase
        model.train()
        running_loss = 0.0
        correct_train_preds = 0
        total_train_preds = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs
            labels = labels

            optimizer.zero_grad()  # Zero gradients for every batch!
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)  # Computes the maximum value across dimension 1 (class dimension)
            correct_train_preds += torch.sum(preds == labels.data)
            total_train_preds += labels.size(0)

        epoch_loss = running_loss / total_train_preds
        epoch_acc = correct_train_preds.double() / total_train_preds

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.4f}')

        

    return train_losses, train_accuracies