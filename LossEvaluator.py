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
    sum_correct_foot_classifications: float
    sum_com_acc_squared_error: np.ndarray
    sum_contact_forces_squared_error: np.ndarray
    confusion_matrix: np.ndarray

    def __init__(self, contact_weight=1.0, com_acc_weight=1.0, contact_forces_weight=1.0):
        self.contact_loss = torch.nn.CrossEntropyLoss()
        self.com_acc_criterion = torch.nn.MSELoss()
        self.contact_forces_criterion = torch.nn.MSELoss()
        self.contact_weight = contact_weight
        self.com_acc_weight = com_acc_weight
        self.contact_forces_weight = contact_forces_weight

        self.num_evaluations = 0
        self.sum_loss = 0.0
        self.sum_timesteps = 0
        self.sum_correct_foot_classifications = 0.0
        self.sum_com_acc_mpss_error = np.zeros(3)
        self.sum_contact_forces_N_error = np.zeros(6)
        self.confusion_matrix = np.zeros((4,4), dtype=np.int64)

    def __call__(self, outputs: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Compute the loss
        loss = self.contact_weight * self.contact_loss(outputs[OutputDataKeys.CONTACT], labels[OutputDataKeys.CONTACT])
            # self.com_acc_weight * self.com_acc_criterion(outputs[OutputDataKeys.COM_ACC], labels[OutputDataKeys.COM_ACC]) + \
            # self.contact_forces_weight * self.contact_forces_criterion(
            #     outputs[OutputDataKeys.CONTACT_FORCES], labels[OutputDataKeys.CONTACT_FORCES])

        # Keep track of various performance metrics to report
        with torch.no_grad():
            self.num_evaluations += 1
            _, predicted_clasess = torch.max(outputs[OutputDataKeys.CONTACT], -1)  # get the index of the max probability
            _, label_classes = torch.max(labels[OutputDataKeys.CONTACT], -1)  # get the index of the one hot labels
            sum_correct = torch.sum(predicted_clasess == label_classes, dim=(0)).item()
            timesteps = outputs[OutputDataKeys.CONTACT].shape[0]
            # Add to the confusion matrix
            for i in range(4):
                for j in range(4):
                    self.confusion_matrix[i,j] += torch.sum((predicted_clasess == i) & (label_classes == j), dim=(0)).item()
            self.sum_correct_foot_classifications += sum_correct  # calculate the number of correct predictions
            self.sum_timesteps += timesteps
            self.sum_loss += loss.item()

        return loss

    def print_report(self):
        print('\tLoss: ' + str(self.sum_loss / self.num_evaluations))
        print('\tFoot classification accuracy: ' +
              str(self.sum_correct_foot_classifications / self.sum_timesteps))
        print('\tTrue label frequency:')
        print('\t\tF = '+str(sum(self.confusion_matrix[:,0]))+' ('+str(100*sum(self.confusion_matrix[:,0]) / max(self.confusion_matrix.sum(),1))+'%)')
        print('\t\tL = '+str(sum(self.confusion_matrix[:,1]))+' ('+str(100*sum(self.confusion_matrix[:,1]) / max(self.confusion_matrix.sum(),1))+'%)')
        print('\t\tR = '+str(sum(self.confusion_matrix[:,2]))+' ('+str(100*sum(self.confusion_matrix[:,2]) / max(self.confusion_matrix.sum(),1))+'%)')
        print('\t\tD = '+str(sum(self.confusion_matrix[:,3]))+' ('+str(100*sum(self.confusion_matrix[:,3]) / max(self.confusion_matrix.sum(),1))+'%)')
        print('\tFoot classification confusion matrix (row = prediction, col = correct) [F, L, R, D]: ')
        print('\t\t' + np.array2string(self.confusion_matrix).strip().replace('\n', '\n\t\t'))
        # print('\tCOM acc avg m/s^2 error (per axis): ' +
        #       str(self.sum_com_acc_mpss_error / self.sum_timesteps))
        # print('\tContact force avg N error (per axis), foot 1: ' +
        #       str(self.sum_contact_forces_N_error[:3] / self.sum_timesteps))
        # print('\tContact force avg N error (per axis), foot 2: ' +
        #       str(self.sum_contact_forces_N_error[3:] / self.sum_timesteps))

        # Reset
        self.num_evaluations = 0
        self.sum_loss = 0.0
        self.sum_timesteps = 0
        self.sum_correct_foot_classifications = 0.0
        self.sum_com_acc_mpss_error = np.zeros(3)
        self.sum_contact_forces_N_error = np.zeros(6)
        self.confusion_matrix = np.zeros((4,4), dtype=np.int64)

        pass
