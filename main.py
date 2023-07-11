import torch
from torch.utils.data import DataLoader
from AddBiomechanicsDataset import AddBiomechanicsDataset, InputDataKeys, OutputDataKeys
from models.FeedForwardBaseline import FeedForwardBaseline
from models.TransformerBaseline import TransformerBaseline
from LossEvaluator import LossEvaluator
from typing import Dict, Tuple, List

# The window size is the number of frames we want to have as context for our model to make predictions.
window_size = 10
# The number of timesteps to skip between each frame in a given window. Data is currently all sampled at 100 Hz, so
# this means 0.2 seconds between each frame. This times window_size is the total time span of each window, which is
# currently 2.0 seconds.
stride = 20
# The batch size is the number of windows we want to load at once, for parallel training and inference on a GPU
batch_size = 32
# The number of epochs is the number of times we want to iterate over the entire dataset during training
epochs = 40
# Learning rate
learning_rate = 1e-3

# Input dofs to train on
input_dofs = ['knee_angle_l', 'knee_angle_r', 'ankle_angle_l', 'ankle_angle_r']

# Create an instance of the dataset
train_dataset = AddBiomechanicsDataset(
    './data/train', window_size, stride, input_dofs=input_dofs, device=torch.device('cpu'))
dev_dataset = AddBiomechanicsDataset(
    './data/dev', window_size, stride, input_dofs=input_dofs, device=torch.device('cpu'))

# Create a DataLoader to load the data in batches
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True)

# Define the model
model = FeedForwardBaseline(len(input_dofs), window_size)


# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    # Iterate over the entire training dataset
    loss_evaluator = LossEvaluator(
        contact_weight=1.0, com_acc_weight=1e-3, contact_forces_weight=1e-3)
    for i, batch in enumerate(train_dataloader):
        inputs: Dict[str, torch.Tensor]
        labels: Dict[str, torch.Tensor]
        inputs, labels = batch

        if i % 100 == 0:
            print('  - Batch '+str(i)+'/'+str(len(train_dataloader)))

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
        for i, batch in enumerate(dev_dataloader):
            if i % 100 == 0:
                print('  - Dev Batch ' + str(i) + '/' + str(len(dev_dataloader)))
            inputs: Dict[str, torch.Tensor]
            labels: Dict[str, torch.Tensor]
            inputs, labels = batch
            outputs = model(inputs)
            loss = dev_loss_evaluator(outputs, labels)
    # Report dev loss on this epoch
    print('Dev Set Evaluation: ')
    dev_loss_evaluator.print_report()
