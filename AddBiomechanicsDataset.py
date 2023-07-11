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
    COM_ACC = 'com_acc'


class OutputDataKeys:
    CONTACT = 'contact'
    COM_ACC = 'com_acc'
    CONTACT_FORCES = 'contact_forces'


class AddBiomechanicsDataset(Dataset):
    folder_path: str
    window_size: int
    stride: int
    device: torch.device
    subjects: List[nimble.biomechanics.SubjectOnDisk]
    input_dofs: List[str]
    input_dof_indices: List[int]
    windows: List[Tuple[nimble.biomechanics.SubjectOnDisk, int, int, str]]

    def __init__(self, folder_path: str, window_size: int, stride: int, input_dofs: List[str], device: torch.device = torch.device('cpu')):
        self.folder_path = folder_path
        self.window_size = window_size
        self.stride = stride
        self.input_dofs = input_dofs
        self.device = device
        self.subjects = []
        self.windows = []

        # Walk the folder path, and check for any with the ".bin" extension (indicating that they are AddBiomechanics binary data files)
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".bin"):
                    # Create a subject object for each file. This will load just the header from this file, and keep that around in memory
                    subject_path = os.path.join(root, file)
                    subject = nimble.biomechanics.SubjectOnDisk(
                        subject_path)
                    # Add the subject to the list of subjects
                    self.subjects.append(subject)
                    # Also, count how many random windows we could select from this subject
                    for trial in range(subject.getNumTrials()):
                        trial_length = subject.getTrialLength(trial)
                        for window_start in range(max(trial_length - (window_size * stride) + 1, 0)):
                            self.windows.append(
                                (subject, trial, window_start, subject_path))

        # Read the dofs from the first subject (assuming they are all the same)
        self.input_dof_indices = []
        skel = self.subjects[0].readSkel()
        for i in range(skel.getNumDofs()):
            dof_name = skel.getDofByIndex(i).getName()
            if dof_name in input_dofs:
                self.input_dof_indices.append(i)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index: int):
        subject, trial, start_frame, subject_path = self.windows[index]
        frames: List[nimble.biomechanics.Frame] = subject.readFrames(
            trial, start_frame, self.window_size, self.stride)
        dt = subject.getTrialTimestep(trial)

        # Convert the frames to a dictionary of matrices, where columns are timesteps and rows are degrees of freedom / dimensions
        # (the DataLoader will then convert this to a batched tensor)

        # Set the random seed to the index, so noise is exactly reproducible every time we retrieve this frame of data
        np.random.seed(index)

        # We first assemble the data into numpy arrays, and then convert to tensors, to save from spurious memory copies which slow down data loading
        numpy_input_dict: Dict[str, np.ndarray] = {}
        numpy_output_dict: Dict[str, np.ndarray] = {}


        poses = [np.copy(frame.pos) for frame in frames]
        numpy_input_dict[InputDataKeys.POS] = np.column_stack([np.array(frame.pos[self.input_dof_indices]) for frame in frames])
        numpy_input_dict[InputDataKeys.VEL] = np.column_stack([np.array(frame.vel[self.input_dof_indices]) for frame in frames])
        numpy_input_dict[InputDataKeys.ACC] = np.column_stack([np.array(frame.acc[self.input_dof_indices]) for frame in frames])
        numpy_input_dict[InputDataKeys.COM_ACC] = np.column_stack([np.array(frame.comAcc) for frame in frames])
        numpy_output_dict[OutputDataKeys.CONTACT] = np.column_stack([np.array(frame.contact, dtype=np.float64) for frame in frames])

        # ###################################################
        # # Plotting
        # import matplotlib.pyplot as plt
        # x = np.arange(self.window_size)
        # # plotting each row
        # for i in range(len(self.input_dofs)):
        #     # plt.plot(x, numpy_input_dict[InputDataKeys.POS][i, :], label='pos_'+self.input_dofs[i])
        #     plt.plot(x, numpy_input_dict[InputDataKeys.VEL][i, :], label='vel_' + self.input_dofs[i])
        #     plt.plot(x, numpy_input_dict[InputDataKeys.ACC][i, :], label='acc_' + self.input_dofs[i])
        # for i in range(3):
        #     plt.plot(x, numpy_input_dict[InputDataKeys.COM_ACC][i, :], label='com_acc_' + str(i))
        # for i in range(2):
        #     plt.plot(x, numpy_output_dict[OutputDataKeys.CONTACT][i, :] * 200 - 100, label='contact_' + str(i))
        # # Add the legend outside the plot
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        # plt.show()
        # ###################################################

        if (numpy_input_dict[InputDataKeys.POS] == 0).all():
            print("Warning: all zeros input POS")
            print('Subject path: '+subject_path)
            print('Trial: '+str(trial))
            print('Trial length: '+str(subject.getTrialLength(trial)))
            print('Start frame: '+str(start_frame))
            print('Window size: '+str(self.window_size))
            for i in range(self.window_size):
                frame = frames[i]
                print('Frame '+str(i)+': ')
                print(frame.pos)
                print('Original copy '+str(i)+': ')
                print(poses[i])

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
