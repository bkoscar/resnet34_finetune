import os
import argparse
import torch
from torch.utils.data import DataLoader
import utils
import split_data
import resnet34_model
from train import Trainer
from test import Tester


def arguments():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Fine-tuning ResNet34")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/exp01_config.json",
        help="Path to the config file",
    )
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--test", action="store_true", help="Test the model")
    parser.add_argument("--epoch", action="store", type=int, help="Epoch to test the model")         
    args = parser.parse_args()
    return args

def run_training(config, model, train_loader, val_loader, device, log_dir, checkpoint_dir):
    trainer = Trainer(config=config, model=model, train_loader=train_loader, val_loader=val_loader, device=device, log_dir=log_dir, checkpoint=checkpoint_dir)
    # Start training
    trainer.train()

def run_testing(config, checkpoint_dir, epoch, classes, device):
    tester = Tester(config=config, checkpoint_dir=checkpoint_dir, epoch=epoch, classes=classes, device=device)
    tester.run()

def main():
    args = arguments()
    config, classes = utils.read_config(args)
    epoch = args.epoch

    # Check if the output directory exists
    main_directory = config["name"]
    log_dir = os.path.join(main_directory, config["output"]["log_dir"])
    checkpoint_dir = os.path.join(main_directory, config["output"]["checkpoint_dir"])

    if not os.path.exists(main_directory):
        os.makedirs(main_directory)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.train:
        # Load datasets
        train, val = split_data.get_data(config, classes)
        train_loader = DataLoader(
            train,
            batch_size=config["train"]["batch_size"],
            shuffle=True,
            num_workers=config["train"]["num_workers"],
        )
        val_loader = DataLoader(
            val,
            batch_size=config["train"]["batch_size"],
            shuffle=False,
            num_workers=config["train"]["num_workers"],
        )
        print(f"INFO: train/val: {len(train_loader.dataset)}/{len(val_loader.dataset)}")
        print("INFO: Training the model")
        model = resnet34_model.get_model()
        utils.check_parameters(model)
        print(f"INFO: Using model {config['model']['model_name']}")
        print(f"INFO: Using device {device}")
        # # Initialize trainer
        run_training(config, model, train_loader, val_loader, device, log_dir, checkpoint_dir)
    elif args.test:
        if args.epoch is None:
            raise ValueError("Please provide an epoch to test the model.")
        print("Testing the model")
        run_testing(config, checkpoint_dir, epoch, classes, device)
        

if __name__ == "__main__":
    main()
    print("Done.")
