import argparse
import json
import os
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.decomposition import TruncatedSVD, DictionaryLearning
from transformers import get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import ReduceLROnPlateau

from analysis.feature_decomposition import decompose_activations
from analysis.utils import get_dict_of_top_k_items

__all__ = ["compute_contrastive_steering_vectors", "train_steering_model"]



import torch
import torch.nn as nn

class SteeringNet(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 100):
        super().__init__()
        self.encoder = nn.Linear(input_size, hidden_size, bias=True)
        self.decoder = nn.Linear(hidden_size, output_size, bias=False)
        self.activ = nn.Tanh()

    def forward(self, x):
        x = self.encoder(x)
        x = self.activ(x)
        x = self.decoder(x)

        return x




class SteeringDataset(Dataset):
    def __init__(self, inputs, targets, dataset_name="pope"):
        assert len(inputs) == len(targets), "Inputs and targets must be the same length"
        self.inputs = [torch.tensor(inp, dtype=torch.float32).squeeze(0) for inp in inputs]
        self.targets = [torch.tensor(tgt, dtype=torch.float32).squeeze(0) for tgt in targets]
        self.dataset_name = dataset_name  # Store dataset name for splitting logic
        self._prepare_dataloaders()       # Automatically prepare loaders after initialization

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
    
    def get_train_shifts(self):
        return np.array([self.targets[i] for i in self.train_indices])

    def _prepare_dataloaders(self):
        total_size = len(self)

        if self.dataset_name == "pope":
            train_ratio = 1100 / 1200
            train_size = int(total_size * train_ratio)
            indices = list(range(total_size))
            self.train_indices = indices[:train_size]
            self.val_indices = indices[train_size:]
        elif "mmsb" in self.dataset_name:
            train_ratio, seed = 0.8, 21 # split generated from (seed=21) used in main mmsb experiments since the very first experiment
            all_idx = np.arange(total_size)
            n_train = int(total_size * train_ratio)
            np.random.seed(seed)
            np.random.shuffle(all_idx)
            self.train_indices = all_idx[:n_train]
            self.val_indices = all_idx[n_train:]
        else:
            raise NotImplementedError(f"Dataset split not implemented for: {self.dataset_name}")

        self.train_dataset = Subset(self, self.train_indices)
        self.val_dataset = Subset(self, self.val_indices)

        self.train_loader = DataLoader(self.train_dataset, batch_size=64, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=64, shuffle=False)

class LearnableSteering:
    def __init__(
        self,
        pos_path: str,
        neg_path: str,
        module: str,
        shift_type: str,
        save_dir: str,
        save_name: str,
        model_name: str,
        input_module: str = None,
        cxt_path: str = None,
        model_class: Any = None,
        logger: Callable = None,
        args: argparse.Namespace = None,
    ):
        self.pos_path = pos_path
        self.neg_path = neg_path
        self.cxt_path = cxt_path
        self.module = module
        self.input_module = input_module
        self.shift_type = shift_type
        self.save_dir = save_dir
        self.save_name = save_name
        self.model_name = model_name
        self.model_class = model_class
        self.hidden_size = args.hidden_size
        self.logger = logger

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.steering_file_base = f"{model_name}_{module.split('.')[-1]}_{shift_type}_{save_name}"
        self.args = args

    def compute_contrastive_vectors(self):
        pos_inf = torch.load(self.pos_path, map_location="cpu")
        neg_inf = torch.load(self.neg_path, map_location="cpu")

        try:
            assert np.all(pos_inf["image"] == neg_inf["image"])
        except:
            assert len(pos_inf["hidden_states"]) == len(neg_inf["hidden_states"])
        n_samples = len(pos_inf["hidden_states"])


        pos_hidden_states = torch.stack([
            pos_inf["hidden_states"][i][self.module]["outputs"][self.shift_type].float()
            for i in range(n_samples)
        ])
        neg_hidden_states = torch.stack([
            neg_inf["hidden_states"][i][self.module]["outputs"][self.shift_type].float()
            for i in range(n_samples)
        ])

        # Save per-sample steering vector
        individual_shifts = pos_hidden_states - neg_hidden_states

        if "mmsb" in self.args.dataset_name and self.args.split == "multi":
            # Normalize steering vectors to make average norms for different prompt completions identical
            coeff1 = torch.norm(individual_shifts[:1125], dim=2).mean() / torch.norm(individual_shifts[1125:1422], dim=2).mean()
            coeff2 = torch.norm(individual_shifts[:1125], dim=2).mean() / torch.norm(individual_shifts[1422:1531], dim=2).mean()
            individual_shifts[1125:1422] = coeff1*individual_shifts[1125:1422]
            individual_shifts[1422:1531] = coeff2*individual_shifts[1422:1531]

        torch.save({"steering_vector": individual_shifts},
                   os.path.join(self.save_dir, self.steering_file_base + ".pth"))
        
        self.logger.info(f"Saving individual shift vectors in : {os.path.join(self.save_dir, self.steering_file_base + '.pth')}")


        # Save mean-based vector
        pos_mean = pos_hidden_states.mean(dim=0)
        neg_mean = neg_hidden_states.mean(dim=0)
        mean_shift = pos_mean - neg_mean

        torch.save({"steering_vector": mean_shift.repeat(n_samples, 1)},
                   os.path.join(self.save_dir, self.steering_file_base + "_mean.pth"))
        
        self.logger.info(f"Saving mean shift vectors in : {os.path.join(self.save_dir, self.steering_file_base + '_mean.pth')}")


    def train_model(self):

        dataset_name = None
        if "pope" in self.pos_path:
            dataset_name = "pope"
        elif "mmsb" in self.args.dataset_name:
            dataset_name = "mmsb"
        else:
            NotImplementedError

        input_inf = torch.load(self.cxt_path, map_location="cpu")["hidden_states"]

        output_data = torch.load(
            os.path.join(self.save_dir, self.steering_file_base + ".pth"),
            map_location="cpu"
        )["steering_vector"]

        if "pope" in dataset_name:
            input_data = [
                input_inf[i][self.module]["inputs"]["last_raw_input"]
                for i in range(len(input_inf))
            ]
        elif "mmsb" in dataset_name:
            input_data = [
                input_inf[i][self.input_module]["outputs"]["last_input"]
                for i in range(len(input_inf))
            ]
        steering_dataset = SteeringDataset(input_data, output_data, dataset_name=dataset_name)
        
        input_size, output_size, hidden_size = self.model_class.model_.config.text_config.max_position_embeddings, self.model_class.model_.config.text_config.max_position_embeddings, self.hidden_size
        net = SteeringNet(input_size, output_size, hidden_size).to(self.device)

        steering_trainer = SteeringTrainer(
            steering_dataset,
            dataset_name,
            net,
            self.device,
            hidden_size=hidden_size,
            best_model_path=f"{self.steering_file_base}.pt",
            logger=self.logger,
        )

        steering_trainer.train()


class SteeringTrainer:
    def __init__(
        self,
        dataset,
        dataset_name,
        model,
        device,
        hidden_size=100,
        alpha=1,
        weight_decay=0.0,
        lr=5e-5,
        batch_size=64,
        num_epochs=100,
        model_name="llava",
        best_model_path="best_model.pt",
        logger: Callable = None,
    ):
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.model = model
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.weight_decay = weight_decay
        self.lr = lr
        self.device = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.model_name = model_name
        self.best_model_path = best_model_path
        self.logger = logger

        self.train_loader = self.dataset.train_loader
        self.val_loader = self.dataset.val_loader

        self._init_model_and_optimizer()

        self.rec_loss = nn.MSELoss(reduction="none")
        self.l1_loss = nn.L1Loss(reduction="none")
        self.cos_sim = nn.CosineSimilarity(dim=1)


    def _init_model_and_optimizer(self):
        # SVD initialization
        if "pope" in self.dataset_name:
            model_svd = TruncatedSVD(n_components=self.hidden_size)
            mean_vec = self.dataset.get_train_shifts().mean(axis=0)
            shifted_data = self.dataset.get_train_shifts() - mean_vec
            comp_activ = model_svd.fit_transform(shifted_data)
            init_comp = model_svd.components_
        elif "mmsb" in self.dataset_name:
            model_dl = DictionaryLearning(
                n_components=self.hidden_size,
                positive_code=True,
                fit_algorithm="cd",
                transform_algorithm="lasso_cd",
                max_iter=5000,
            )
            comp_activ = model_dl.fit_transform(self.dataset.get_train_shifts())
            init_comp = model_dl.components_
            self.lr = 1e-4

        #self.model.decoder.weight = nn.Parameter(torch.tensor(init_comp.T, dtype=torch.float32, requires_grad=True).to(self.device))
        self.model.decoder.weight = nn.Parameter(torch.tensor(init_comp.T).to(self.device)*1.0)

        #self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.warmup_steps = 5 * len(self.train_loader)
        self.total_steps = self.num_epochs * len(self.train_loader)

        if "pope" in self.dataset_name:
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.total_steps
            )
            self.scheduler_1 = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    def train(self):

        best_val_loss = float('inf')

        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            total_samples = 0
            cosine_sim_train = 0

            for inputs, targets in self.train_loader:
                
                inputs, targets = inputs.to(self.device).float(), targets.to(self.device).float()
                self.model.zero_grad()

                pred_shift = self.model(inputs)
                smoothed_target = self.alpha * targets + (1 - self.alpha) * pred_shift.detach()

                rec_losses = self.rec_loss(pred_shift.squeeze(1), smoothed_target.squeeze(1)).mean(dim=1)
                l1_losses = self.l1_loss(pred_shift.squeeze(1), smoothed_target.squeeze(1)).mean(dim=1)
                cos_losses = self.cos_sim(pred_shift.squeeze(1), smoothed_target.squeeze(1))
                cosine_weight = min(1.0, epoch / 100) * 0.1
                if "mmsb" in self.dataset_name:
                    cosine_weight = 0.1
                total_losses = rec_losses + l1_losses - cosine_weight * cos_losses

                loss = total_losses.mean()

                if epoch > self.num_epochs / 2 and not "mmsb" in self.dataset_name:
                    top_k = int(0.8 * total_losses.size(0))
                    batch_loss = total_losses.topk(top_k).values.mean()
                else:
                    batch_loss = total_losses.mean()


                batch_loss.backward()
                self.optimizer.step()
                if not "mmsb" in self.dataset_name:
                    self.scheduler.step()

                running_loss += loss.item()
                total_samples += len(targets)
                cosine_sim_train += cos_losses.sum().item()

            train_loss = (running_loss) / (total_samples/self.train_loader.batch_size)

            val_loss, val_cos_sim = self._evaluate(self.val_loader)

            # Scheduler step and model checkpoint
            if not "mmsb" in self.dataset_name:
                self.scheduler_1.step(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), self.best_model_path)
                    self.logger.info(f"✅ New best model saved with val loss: {best_val_loss:.4f} at epoch {epoch + 1}")

            self.logger.info(
                f"Epoch [{epoch+1}/{self.num_epochs}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Train Cos: {cosine_sim_train / total_samples:.4f} | "
                f"Val Cos: {val_cos_sim:.4f}"
            )
        if "mmsb" in self.dataset_name:
            torch.save(self.model.state_dict(), self.best_model_path)
            best_val_loss = min(val_loss, best_val_loss)
            self.logger.info(f"For mmsb dataset, simply save the final model at epoch 100")
            self.logger.info(f"✅ Model saved with val loss: {best_val_loss:.4f} at epoch {epoch + 1}")

    def _evaluate(self, loader):
        self.model.eval()
        total_loss, cosine_sim_total = 0.0, 0.0
        total_samples = 0

        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.float().to(self.device), targets.float().to(self.device)
                pred_shift = self.model(inputs)
                pred_shift, targets = pred_shift.squeeze(1), targets.squeeze(1)

                rec_losses = self.rec_loss(pred_shift, targets).mean(dim=1)
                l1_losses = self.l1_loss(pred_shift, targets).mean(dim=1)
                cos_losses = self.cos_sim(pred_shift, targets)
                total_losses = rec_losses + l1_losses - 0.05 * cos_losses

                batch_loss = total_losses.mean()
                total_loss += batch_loss.item() * inputs.size(0)
                cosine_sim_total += cos_losses.sum().item()
                total_samples += inputs.size(0)

        return total_loss / total_samples, cosine_sim_total / total_samples
