import torch
import torch.nn as nn
from typing import Dict
from AddBiomechanicsDataset import InputDataKeys, OutputDataKeys


class FeedForwardBaseline(nn.Module):
    dofs: int
    window_size: int
    hidden_size: int

    def __init__(self, dofs: int, window_size: int, hidden_size: int = 64):
        super(FeedForwardBaseline, self).__init__()
        self.dofs = dofs
        self.window_size = window_size
        self.hidden_size = hidden_size

        # Compute input and output sizes

        # For input, we need each dof, for position and velocity and acceleration, for each frame in the window, and then also the COM position, velocity, and acceleration for each frame in the window
        input_size = (dofs * window_size * 3) + (3 * window_size * 3)
        # For output, we have two foot-ground contact (2), COM acceleration (3), and contact forces (3 vector force * 2 feet = 6)
        output_size = (2 + 3 + 6) * window_size

        self.fc1 = nn.Linear(input_size, hidden_size, dtype=torch.float64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size, dtype=torch.float64)
        self.contact_sigmoid = nn.Sigmoid()

    def forward(self, input: Dict[str, torch.Tensor]):
        # Get the position, velocity, and acceleration tensors
        flattened_poses = input[InputDataKeys.POS].flatten(start_dim=-2)
        flattened_vels = input[InputDataKeys.VEL].flatten(start_dim=-2)
        flattened_accs = input[InputDataKeys.ACC].flatten(start_dim=-2)
        flattened_com_poses = input[InputDataKeys.COM_POS].flatten(
            start_dim=-2)
        flattened_com_vels = input[InputDataKeys.COM_VEL].flatten(start_dim=-2)
        flattened_com_accs = input[InputDataKeys.COM_ACC].flatten(start_dim=-2)
        # Concatenate the tensors along the last dimension to get a single input vector
        flattened_input = torch.cat((flattened_poses, flattened_vels, flattened_accs,
                                    flattened_com_poses, flattened_com_vels, flattened_com_accs), dim=-1)
        # Actually run the forward pass
        x = self.fc1(flattened_input)
        x = self.relu(x)
        x = self.fc2(x)
        # Now we need to split the output into the different components
        output_dict: Dict[str, torch.Tensor] = {}
        # First, the contact predictions
        output_dict[OutputDataKeys.CONTACT] = self.contact_sigmoid(x[:, :2*self.window_size]).reshape(
            (-1, 2, self.window_size)
        )
        # Then the COM acceleration
        output_dict[OutputDataKeys.COM_ACC] = x[:, 2*self.window_size:5*self.window_size].reshape(
            (-1, 3, self.window_size)
        )
        # Then the contact forces
        output_dict[OutputDataKeys.CONTACT_FORCES] = x[:, 5*self.window_size:].reshape(
            (-1, 6, self.window_size)
        )
        return output_dict
