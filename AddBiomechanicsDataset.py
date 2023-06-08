import nimblephysics as nimble
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple
import os
import numpy as np


class InputDataKeys:
    POS = 'pos'
    VEL = 'vel'
    ACC = 'acc'
    COM_POS = 'com_pos'
    COM_VEL = 'com_vel'
    COM_ACC = 'com_acc'


class OutputDataKeys:
    CONTACT = 'contact'
    COM_ACC = 'com_acc'
    CONTACT_FORCES = 'contact_forces'


class AddBiomechanicsDataset(Dataset):
    folder_path: str
    window_size: int
    position_noise: float
    device: torch.device
    subjects: List[nimble.biomechanics.SubjectOnDisk]
    windows: List[Tuple[nimble.biomechanics.SubjectOnDisk, int, int]]
    dofs: int

    def __init__(self, folder_path: str, window_size: int, position_noise: float = 0.0, device: torch.device = torch.device('cpu')):
        self.folder_path = folder_path
        self.window_size = window_size
        self.position_noise = position_noise
        self.device = device
        self.subjects = []
        self.windows = []
        self.dofs = 0

        # Walk the folder path, and check for any with the ".bin" extension (indicating that they are AddBiomechanics binary data files)
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".bin"):
                    # Create a subject object for each file. This will load just the header from this file, and keep that around in memory
                    subject = nimble.biomechanics.SubjectOnDisk(
                        os.path.join(root, file))
                    # Add the subject to the list of subjects
                    self.subjects.append(subject)
                    # Also, count how many random windows we could select from this subject
                    for trial in range(subject.getNumTrials()):
                        trial_length = subject.getTrialLength(trial)
                        for window_start in range(max(trial_length - window_size + 1, 0)):
                            self.windows.append(
                                (subject, trial, window_start))

        # Read the dofs from the first subject (assuming they are all the same)
        self.dofs = self.subjects[0].getNumDofs()

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index: int):
        subject, trial, start_frame = self.windows[index]
        frames: List[nimble.biomechanics.Frame] = subject.readFrames(
            trial, start_frame, self.window_size)
        dt = subject.getTrialTimestep(trial)

        # Convert the frames to a dictionary of matrices, where columns are timesteps and rows are degrees of freedom / dimensions
        # (the DataLoader will then convert this to a batched tensor)

        # Set the random seed to the index, so noise is exactly reproducible every time we retrieve this frame of data
        np.random.seed(index)

        # We first assemble the data into numpy arrays, and then convert to tensors, to save from spurious memory copies which slow down data loading
        numpy_input_dict: Dict[str, np.ndarray] = {}
        numpy_output_dict: Dict[str, np.ndarray] = {}

        numpy_input_dict[InputDataKeys.POS] = np.random.randn(
            self.dofs, self.window_size) * self.position_noise
        numpy_input_dict[InputDataKeys.VEL] = np.random.randn(
            self.dofs, self.window_size) * self.position_noise * (1.0 / dt)  # Finite differencing amplifies noise
        numpy_input_dict[InputDataKeys.ACC] = np.random.randn(
            self.dofs, self.window_size) * self.position_noise * (1.0 / dt) * (1.0 / dt)  # Finite differencing twice amplifies noise twice
        numpy_input_dict[InputDataKeys.COM_POS] = np.random.randn(
            3, self.window_size) * self.position_noise
        numpy_input_dict[InputDataKeys.COM_VEL] = np.random.randn(
            3, self.window_size) * self.position_noise * (1.0 / dt)  # Finite differencing amplifies noise
        numpy_input_dict[InputDataKeys.COM_ACC] = np.random.randn(
            3, self.window_size) * self.position_noise * (1.0 / dt) * (1.0 / dt)  # Finite differencing twice amplifies noise twice

        numpy_output_dict[OutputDataKeys.CONTACT] = np.zeros(
            (2, self.window_size))
        numpy_output_dict[OutputDataKeys.COM_ACC] = np.zeros(
            (3, self.window_size))
        numpy_output_dict[OutputDataKeys.CONTACT_FORCES] = np.zeros(
            (6, self.window_size))

        for i in range(self.window_size):
            frame = frames[i]
            numpy_input_dict[InputDataKeys.POS][6:, i] += frame.pos[6:]
            numpy_input_dict[InputDataKeys.VEL][:, i] += frame.vel
            numpy_input_dict[InputDataKeys.ACC][:, i] += frame.acc
            # numpy_input_dict[InputDataKeys.COM_POS][:, i] += frame.comPos
            numpy_input_dict[InputDataKeys.COM_VEL][:, i] += frame.comVel
            numpy_input_dict[InputDataKeys.COM_ACC][:, i] += frame.comAcc
            numpy_output_dict[OutputDataKeys.CONTACT][:, i] = frame.contact
            numpy_output_dict[OutputDataKeys.COM_ACC][:, i] = frame.comAcc
            numpy_output_dict[OutputDataKeys.CONTACT_FORCES][:,
                                                             i] = frame.groundContactForce

        # Doing things inside torch.no_grad() suppresses warnings and gradient tracking
        with torch.no_grad():
            input_dict: Dict[str, torch.Tensor] = {}
            for key in numpy_input_dict:
                input_dict[key] = torch.tensor(
                    numpy_input_dict[key], device=self.device)

            label_dict: Dict[str, torch.Tensor] = {}
            for key in numpy_output_dict:
                label_dict[key] = torch.tensor(
                    numpy_output_dict[key], device=self.device)

        # Return the input and output dictionaries at this timestep

        return input_dict, label_dict
