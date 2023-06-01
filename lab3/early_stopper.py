import torch
import numpy as np


class EarlyStopper:
    def __init__(self, model, save_path, patience=1, min_delta=0):
        self.model = model
        self.save_path = save_path
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            torch.save(self.model.state_dict(), self.save_path)
            print('best model!')
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
