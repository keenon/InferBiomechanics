import torch
from torch.utils.data import DataLoader
from AddBiomechanicsDataset import AddBiomechanicsDataset, InputDataKeys, OutputDataKeys
from models.FeedForwardBaseline import FeedForwardBaseline
from typing import Dict, Tuple

# The window size is the number of frames we want to have as context for our model to make predictions
window_size = 10
# The batch size is the number of windows we want to load at once, for parallel training and inference on a GPU
batch_size = 32
# The number of epochs is the number of times we want to iterate over the entire dataset during training
epochs = 10
# Learning rate
learning_rate = 1e-3

# Create an instance of the dataset
train_dataset = AddBiomechanicsDataset(
    './data/train', window_size, position_noise=0.01, device=torch.device('cpu'))
dev_dataset = AddBiomechanicsDataset(
    './data/dev', window_size, position_noise=0.01, device=torch.device('cpu'))

# Create a DataLoader to load the data in batches
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True)

# Define the model
model = FeedForwardBaseline(train_dataset.dofs, window_size)

# Define the loss function
contact_criterion = torch.nn.BCEWithLogitsLoss()
com_acc_criterion = torch.nn.MSELoss()
contact_forces_criterion = torch.nn.MSELoss()

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    # Iterate over the entire training dataset
    sum_train_loss = 0.0
    for batch in train_dataloader:
        inputs: Dict[str, torch.Tensor]
        labels: Dict[str, torch.Tensor]
        inputs, labels = batch

        # Clear the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute the loss as the sum of all three criteria (contact accuracy, COM acceleration MSE, and contact forces MSE)
        loss = contact_criterion(outputs[OutputDataKeys.CONTACT], labels[OutputDataKeys.CONTACT]) + \
            com_acc_criterion(outputs[OutputDataKeys.COM_ACC], labels[OutputDataKeys.COM_ACC]) + \
            contact_forces_criterion(
                outputs[OutputDataKeys.CONTACT_FORCES], labels[OutputDataKeys.CONTACT_FORCES])
        with torch.no_grad():
            sum_train_loss += loss.item()

        # Backward pass
        loss.backward()

        # Update the model's parameters
        optimizer.step()
    # Report training loss on this epoch
    print('Epoch '+str(epoch)+' training loss: ' +
          str(sum_train_loss/len(dev_dataloader)))

    # At the end of each epoch, evaluate the model on the dev set
    sum_dev_loss = 0.0
    with torch.no_grad():
        for batch in dev_dataloader:
            inputs: Dict[str, torch.Tensor]
            labels: Dict[str, torch.Tensor]
            inputs, labels = batch
            outputs = model(inputs)
            loss = contact_criterion(outputs[OutputDataKeys.CONTACT], labels[OutputDataKeys.CONTACT]) + \
                com_acc_criterion(outputs[OutputDataKeys.COM_ACC], labels[OutputDataKeys.COM_ACC]) + \
                contact_forces_criterion(
                    outputs[OutputDataKeys.CONTACT_FORCES], labels[OutputDataKeys.CONTACT_FORCES])
            sum_dev_loss += loss.item()
    # Report dev loss on this epoch
    print('Epoch '+str(epoch)+' dev loss: ' +
          str(sum_dev_loss/len(dev_dataloader)))
