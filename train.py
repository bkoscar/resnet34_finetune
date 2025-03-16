import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import numpy as np 
import random
import tqdm
import dataset
import utils

class Trainer:
    def __init__(
        self, config, model, train_loader, val_loader, device, log_dir, checkpoint
    ):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_workers = config["train"]["num_workers"]
        self.epochs = config["train"]["num_epochs"]
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.metrics = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config["train"]["lr"],
            weight_decay=config["train"]["weight_decay"],
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=10
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = device
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.model.to(self.device)

        # seed
        seed = config["train"]["seed"]
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Early stopping parameters
        self.patience = config["train"]["early_stopping"]["patience"]  
        self.min_delta = config["train"]["early_stopping"]['min_delta'] 
        self.early_stop_counter = 0  

    def train_epoch(self, epoch):
        progress_bar = tqdm.tqdm(
            enumerate(self.train_loader), total=len(self.train_loader)
        )
        self.model.train()
        running_loss = 0.0
        for i, data in progress_bar:
            inputs, labels = data["img"].to(self.device), data["target"].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix({"loss": running_loss / (i + 1)})
        avg_loss = running_loss / len(self.train_loader)
        self.writer.add_scalar("Loss/train", avg_loss, epoch)

        img_grid = torchvision.utils.make_grid(inputs)
        self.writer.add_image(f"Train Images/Epoch {epoch+1}", img_grid, epoch)
        return avg_loss

    def val_epoch(self, epoch):
        progress_bar = tqdm.tqdm(enumerate(self.val_loader), total=len(self.val_loader))
        self.model.eval()
        running_loss = 0.0
        all_preds, all_labels = [], []
        log_inputs = None
        log_labels = None
        log_preds = None
        with torch.no_grad():
            for i, data in progress_bar:
                inputs, labels = data["img"].to(self.device), data["target"].to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                if i == 0:
                    log_inputs = inputs.clone()
                    log_labels = labels.clone()
                    log_preds = preds.clone()
                progress_bar.set_postfix({"loss": running_loss / (i + 1)})
        avg_loss = running_loss / len(self.val_loader)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted")
        self.writer.add_scalar("Loss/val", avg_loss, epoch)
        self.writer.add_scalar("Metrics/val_acc", acc, epoch)
        self.writer.add_scalar("Metrics/val_f1", f1, epoch)
        if log_inputs is not None:
            img_grid = torchvision.utils.make_grid(log_inputs)
            self.writer.add_image(f"Validation Images/Epoch {epoch+1}", img_grid, epoch)
            for j in range(len(log_inputs)):
                self.writer.add_text(
                    f"Validation Predictions/Epoch {epoch+1}",
                    f"Image {j}: True Label: {log_labels[j].item()}, Predicted: {log_preds[j].item()}",
                    epoch,
                )

        return avg_loss, acc, f1

    def train(self):
        for epoch in range(self.epochs):
            train_loss = self.train_epoch(epoch)
            print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}")
            val_loss, val_acc, val_f1 = self.val_epoch(epoch)
            print(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
            self.metrics["train_loss"].append(train_loss)
            self.metrics["val_loss"].append(val_loss)
            self.metrics["val_acc"].append(val_acc)
            self.metrics["val_f1"].append(val_f1)
            self.scheduler.step(val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(
                    self.model.state_dict(),
                    f"{self.checkpoint_dir}/best_model_epoch_{epoch+1}.pth",
                )

            # Early stopping: check
            if val_acc > self.best_val_acc + self.min_delta and val_f1 > self.best_val_f1 + self.min_delta:
                self.best_val_acc = val_acc
                self.best_val_f1 = val_f1
                self.early_stop_counter = 0  # Resetear el contador si las métricas mejoran
            else:
                self.early_stop_counter += 1

            if self.early_stop_counter >= self.patience:
                print(f"Early stopping at epoch {epoch+1} - No improvement in accuracy or F1 score.")
                break  

        self.writer.close()
        return self.metrics
