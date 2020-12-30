from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

class Trainer():
    def __init__(self, model, optimizer, crit):
        self.model = model
        self.optimizer = optimizer
        self.crit = crit

        super().__init__()

    def _train(self, train_loader, config):
        self.model.train()
        total_loss = 0
        for (x_i, y_i) in train_loader:
            if config.gpu_id >= 0:
                x_i = x_i.cuda(config.gpu_id)
                y_i = y_i.cuda(config.gpu_id)
            y_hat_i = self.model(x_i)
            loss_i = self.crit(y_hat_i, y_i)
            self.optimizer.zero_grad()
            loss_i.backward()
            self.optimizer.step()
            # prevent memory leak
            total_loss += float(loss_i)
        return total_loss / len(train_loader)
    
    def _validate(self, val_loader, config):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            for (x_i, y_i) in val_loader:
                if config.gpu_id >= 0:
                    x_i = x_i.cuda(config.gpu_id)
                    y_i = y_i.cuda(config.gpu_id)    
                y_hat_i = self.model(x_i)
                loss_i = self.crit(y_hat_i, y_i)
                total_loss += float(loss_i)
            return total_loss / len(val_loader)

    def train(self, train_loader, val_loader, config):
        lowest_loss = np.inf
        best_model = None
        for epoch_index in range(config.n_epochs):
            train_loss = self._train(train_loader, config)
            valid_loss = self._validate(val_loader, config)
            if valid_loss <= lowest_loss:
                lowest_loss = valid_loss
                best_model = deepcopy(self.model.state_dict())
            print(f'Epoch{epoch_index+1}/{config.n_epochs}: train_loss={train_loss:.3f}, valid_loss={valid_loss:.3f}')
        self.model.load_state_dict(best_model)
        return self.model