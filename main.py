import torch
from torch.utils.data import DataLoader
from AddBiomechanicsDataset import AddBiomechanicsDataset, InputDataKeys, OutputDataKeys
from models.FeedForwardBaseline import FeedForwardBaseline
from models.TransformerBaseline import TransformerBaseline
from LossEvaluator import LossEvaluator
from typing import Dict, Tuple, List

# The window size is the number of frames we want to have as context for our model to make predictions
window_size = 10
# The batch size is the number of windows we want to load at once, for parallel training and inference on a GPU
batch_size = 32
# The number of epochs is the number of times we want to iterate over the entire dataset during training
epochs = 40
# Learning rate
learning_rate = 1e-3
# Noise to artificially add to the position data during training / testing
position_noise = 0.0

# Create an instance of the dataset
train_dataset = AddBiomechanicsDataset(
    './data/train', window_size, position_noise=position_noise, device=torch.device('cpu'))
dev_dataset = AddBiomechanicsDataset(
    './data/dev', window_size, position_noise=position_noise, device=torch.device('cpu'))

# Create a DataLoader to load the data in batches
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True)

# Define the model
model = TransformerBaseline(train_dataset.dofs, window_size, num_layers=1)


# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    # Iterate over the entire training dataset
    loss_evaluator = LossEvaluator(
        contact_weight=1.0, com_acc_weight=1e-3, contact_forces_weight=1e-3)
    for batch in train_dataloader:
        inputs: Dict[str, torch.Tensor]
        labels: Dict[str, torch.Tensor]
        inputs, labels = batch

        # Clear the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute the loss
        loss = loss_evaluator(outputs, labels)

        # Backward pass
        loss.backward()

        # Update the model's parameters
        optimizer.step()
    # Report training loss on this epoch
    print('Epoch '+str(epoch)+': ')
    print('Training Set Evaluation: ')
    loss_evaluator.print_report()

    # At the end of each epoch, evaluate the model on the dev set
    dev_loss_evaluator = LossEvaluator(
        contact_weight=1.0, com_acc_weight=1e-3, contact_forces_weight=1e-3)
    with torch.no_grad():
        for batch in dev_dataloader:
            inputs: Dict[str, torch.Tensor]
            labels: Dict[str, torch.Tensor]
            inputs, labels = batch
            outputs = model(inputs)
            loss = dev_loss_evaluator(outputs, labels)
    # Report dev loss on this epoch
    print('Dev Set Evaluation: ')
    dev_loss_evaluator.print_report()
