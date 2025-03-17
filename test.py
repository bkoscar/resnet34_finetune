import torch
import numpy as np
import json
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader

import resnet34_model
import utils
import split_data
import dataset


class Tester:

    def __init__(self, config, checkpoint_dir, epoch, classes, device):
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.epoch = epoch
        self.metadata = utils.read_metadata(config["data"]["metadata_file"])
        self.model_name = self.checkpoint_dir + f"/best_model_epoch_{self.epoch}.pth"
        print(f"INFO: Using model {self.model_name}")
        self.classes = classes
        self.device = device
        self.model = resnet34_model.get_model(num_classes=len(classes))
        # Load the model weights
        checkpoint = torch.load(self.model_name, map_location=device)
        self.model.load_state_dict(checkpoint)
        self.model.to(device)
        self.model.eval()

        self.main_directory = config['name']
        self.results_directory = os.path.join(self.main_directory, "results")
        os.makedirs(self.results_directory, exist_ok=True)

    def scored_metadata(self):
        self.metadata["predictions"] = -1
        self.metadata["cross_entropy_loss"] = float(np.nan)
        self.metadata["logits"] = float(np.nan)
        image_dataset = dataset.ImageDataset(
            config=self.config,
            metadata=self.metadata,
            classes=self.classes,
            transform=True,
        )
        dataloader = DataLoader(
            image_dataset, batch_size=self.config["train"]["batch_size"], shuffle=False
        )

        criterion = torch.nn.CrossEntropyLoss()

        all_predictions = []
        all_losses = []
        all_logits = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Processing batches"):
                inputs = batch["img"].to(self.device)
                targets = batch["target"].to(self.device)
                logits = self.model(inputs)
                loss = criterion(logits, targets)

                predictions = torch.argmax(logits, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_losses.extend([round(loss.item(), 5)] * len(predictions))
                all_logits.extend(np.round(logits.cpu().numpy(), 5))

        self.metadata["predictions"] = all_predictions
        self.metadata["cross_entropy_loss"] = all_losses
        self.metadata["logits"] = all_logits
        scored_metadata_path = os.path.join(self.results_directory, "scored_metadata.csv")
        self.metadata.to_csv(scored_metadata_path, index=False)
        print(f"INFO: Saved scored metadata to {scored_metadata_path}")
        return self.metadata

    def calculate_metrics(self):
        metrics = {}
        confusion_matrices = {}
        unique_split = self.metadata["split"].unique().tolist()
        for split in unique_split:
            split_metadata = self.metadata[self.metadata["split"] == split]
            y_true = split_metadata["source_label"].map(self.classes)
            y_pred = split_metadata["predictions"]
            accuracy = accuracy_score(y_true, y_pred)
            report = classification_report(y_true, y_pred, output_dict=True)
            avg_loss = round(split_metadata["cross_entropy_loss"].mean(), 5)
            metrics[split] = {
                "accuracy_score": accuracy,
                "classification_report": report,
                "average_loss": avg_loss
            }
            confusion_matrices[split] = confusion_matrix(y_true, y_pred)
        metrics_path = os.path.join(self.results_directory, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to {metrics_path}")
        self.plot_metrics(confusion_matrices)

    def plot_metrics(self, confusion_matrices):
        for split, cm in confusion_matrices.items():
            plt.figure(figsize=(10, 7))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.classes, yticklabels=self.classes)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix for {split}')
            confusion_matrix_path = os.path.join(self.results_directory, f'confusion_matrix_{split}.png')
            plt.savefig(confusion_matrix_path)
            plt.close()

        for split in self.metadata["split"].unique().tolist():
            split_metadata = self.metadata[self.metadata["split"] == split]
            plt.figure(figsize=(10, 7))
            ax = sns.countplot(x=split_metadata["source_label"], order=self.classes)
            plt.xlabel('Class')
            plt.ylabel('Count')
            plt.title(f'Data Distribution for {split}')
            for p in ax.patches:
                ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='center', fontsize=12, color='black', xytext=(0, 5), 
                            textcoords='offset points')
            data_distribution_path = os.path.join(self.results_directory, f'data_distribution_{split}.png')
            plt.savefig(data_distribution_path)
            plt.close()

    def run(self):
        if not os.path.exists("scored_metadata.csv"):
            try:
                self.scored_metadata()
            except Exception as e:
                print(f"An error occurred while scoring metadata: {e}")
                return
        else:
            self.metadata = utils.read_metadata("scored_metadata.csv")
            print("INFO: Loaded existing scored metadata from scored_metadata.csv")

        try:
            self.calculate_metrics()
        except Exception as e:
            print(f"An error occurred while calculating metrics: {e}")