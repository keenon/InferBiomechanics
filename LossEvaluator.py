import torch
from torch.utils.data import DataLoader
from AddBiomechanicsDataset import AddBiomechanicsDataset, InputDataKeys, OutputDataKeys
from models.FeedForwardBaseline import FeedForwardBaseline
from typing import Dict, Tuple, List
import numpy as np


class LossEvaluator:
    num_evaluations: int
    sum_loss: float
    sum_timesteps: int
    sum_correct_foot_classifications: np.ndarray
    sum_com_acc_squared_error: np.ndarray
    sum_contact_forces_squared_error: np.ndarray

    def __init__(self, contact_weight=1.0, com_acc_weight=1.0, contact_forces_weight=1.0):
        self.contact_criterion = torch.nn.BCELoss()
        self.com_acc_criterion = torch.nn.MSELoss()
        self.contact_forces_criterion = torch.nn.MSELoss()
        self.contact_weight = contact_weight
        self.com_acc_weight = com_acc_weight
        self.contact_forces_weight = contact_forces_weight

        self.num_evaluations = 0
        self.sum_loss = 0.0
        self.sum_timesteps = 0
        self.sum_correct_foot_classifications = np.zeros(2)
        self.sum_com_acc_mpss_error = np.zeros(3)
        self.sum_contact_forces_N_error = np.zeros(6)

    def __call__(self, outputs: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Compute the loss
        loss = self.contact_weight * self.contact_criterion(outputs[OutputDataKeys.CONTACT], labels[OutputDataKeys.CONTACT])
            # self.com_acc_weight * self.com_acc_criterion(outputs[OutputDataKeys.COM_ACC], labels[OutputDataKeys.COM_ACC]) + \
            # self.contact_forces_weight * self.contact_forces_criterion(
            #     outputs[OutputDataKeys.CONTACT_FORCES], labels[OutputDataKeys.CONTACT_FORCES])

        # Keep track of various performance metrics to report
        with torch.no_grad():
            self.num_evaluations += 1
            rounded_contact = torch.round(outputs[OutputDataKeys.CONTACT])
            # Sum over time, and over the batch
            self.sum_correct_foot_classifications += torch.sum(
                rounded_contact == labels[OutputDataKeys.CONTACT], dim=(0, 2)).numpy()
            # self.sum_com_acc_mpss_error += torch.sum(
            #     torch.abs(outputs[OutputDataKeys.COM_ACC] - labels[OutputDataKeys.COM_ACC]), dim=(0, 2)).numpy()
            # self.sum_contact_forces_N_error += torch.sum(
            #     torch.abs(outputs[OutputDataKeys.CONTACT_FORCES] - labels[OutputDataKeys.CONTACT_FORCES]), dim=(0, 2)).numpy()
            self.sum_timesteps += outputs[OutputDataKeys.CONTACT].shape[0] * \
                outputs[OutputDataKeys.CONTACT].shape[2]
            self.sum_loss += loss.item()

        return loss

    def print_report(self):
        print('\tLoss: ' + str(self.sum_loss / self.num_evaluations))
        print('\tFoot classification accuracy: ' +
              str(self.sum_correct_foot_classifications / self.sum_timesteps))
        print('\tCOM acc avg m/s^2 error (per axis): ' +
              str(self.sum_com_acc_mpss_error / self.sum_timesteps))
        print('\tContact force avg N error (per axis), foot 1: ' +
              str(self.sum_contact_forces_N_error[:3] / self.sum_timesteps))
        print('\tContact force avg N error (per axis), foot 2: ' +
              str(self.sum_contact_forces_N_error[3:] / self.sum_timesteps))
        pass
