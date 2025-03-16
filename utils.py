import os
import argparse
import json
import shutil
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torchvision.io import read_image
import torchvision.transforms.functional as F
from torchvision import transforms
from sklearn.model_selection import train_test_split


def check_devices():
    """Checks the available devices for computation (MPS, CUDA, or CPU) and prints the current device in use.

    If MPS (Apple GPU) is available, it will use that; if CUDA (NVIDIA GPU) is available, it will use that;
    otherwise, it will fall back to using the CPU. A tensor is created on the selected device to ensure the device is accessible.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        backend = "MPS (Apple GPU)"
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        backend = f"CUDA (NVIDIA GPU: {torch.cuda.get_device_name(0)})"
    else:
        device = torch.device("cpu")
        backend = "CPU"
    x = torch.ones(1, device=device)
    print(f"Using {backend}")
    print(x)


def copy_data(source_path: str, dest_folder: str):
    """Copies image files from a source folder to a destination folder, preserving the 'train' and 'val' subdirectories.

    Args:
        source_path (str): The path to the source directory containing the image files.
        dest_folder (str): The path to the destination directory where the images will be copied.
    """
    subfolders = ["train", "val"]
    os.makedirs(dest_folder, exist_ok=True)
    for subfolder in subfolders:
        subfolder_path = os.path.join(source_path, subfolder)
        if os.path.exists(subfolder_path) and os.path.isdir(subfolder_path):
            for root, _, files in os.walk(subfolder_path):
                for filename in files:
                    if filename.endswith((".jpg")):
                        src = os.path.join(root, filename)
                        dst = os.path.join(dest_folder, filename)
                        shutil.copy(src, dst)
                        print(f"Copiado: {src} → {dst}")


def create_metadata(dataset_folder_path: str):
    """Creates a metadata CSV file from the image files in a dataset folder.

    Args:
        dataset_folder_path (str): The path to the directory containing the dataset images.
    """
    all_files = [
        file
        for file in os.listdir(dataset_folder_path)
        if file.endswith(".jpg")
        and os.path.isfile(os.path.join(dataset_folder_path, file))
    ]
    df = pd.DataFrame(all_files, columns=["file_name"])
    df["source_label"] = df["file_name"].apply(lambda x: x.split("_")[1])
    df.to_csv(
        "/Users/oscar/Documents/Documents-Local/DeepLearning/Datasets/metadata.csv",
        index=False,
    )


def read_config(args):
    """Reads configuration and class metadata from JSON files.

    Args:
        args (argparse): Arguments containing the path to the configuration file.

    Returns:
        tuple: A tuple containing the config dictionary and classes list read from the respective JSON files.
    """
    with open(args.config) as f:
        config = json.load(f)
    with open(config["classes"]) as f:
        classes = json.load(f)
    return config, classes


def read_metadata(metadata_path):
    """_summary_

    Args:
        metadata_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    metadata = pd.read_csv(metadata_path)
    return metadata


def read_img(img_path, normalize=False, show=False, transform=False):
    """_summary_

    Args:
        img_path (_type_): _description_
        normalize (bool, optional): _description_. Defaults to False.
        show (bool, optional): _description_. Defaults to False.
        transform (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    image = read_image(img_path).float() / 255.0  # Leer imagen y convertir a [0,1]

    if normalize:
        mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        image = (image - mean) / std

    if transform:
        transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(p=0.5)]
        )
        image = transform(image)

    if show:
        show_img(
            image, title=f"Imagen: {os.path.basename(img_path)}", unnormalize=normalize
        )

    return image


def show_img(img, title=None, unnormalize=False):
    """_summary_

    Args:
        img (_type_): _description_
        title (_type_, optional): _description_. Defaults to None.
        unnormalize (bool, optional): _description_. Defaults to False.
    """
    if unnormalize:
        mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        img = img * std + mean  # Desnormalizar

    img = F.to_pil_image(img)  # Convertir el tensor en imagen PIL
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()

def split_dataset(config):
    metadata = read_metadata(config["data"]["metadata_file"])
    X = metadata.iloc[:, 0] 
    y = metadata.iloc[:, 1]  
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=config['split_features']['train_split'], random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=config['split_features']['test_split'], random_state=42, stratify=y_temp
    )
    metadata["split"] = "unknown"
    metadata.loc[X_train.index, "split"] = "train"
    metadata.loc[X_val.index, "split"] = "val"
    metadata.loc[X_test.index, "split"] = "test"
    total = len(metadata)
    train_count = len(X_train)
    val_count = len(X_val)
    test_count = len(X_test)
    train_pct = (train_count / total) * 100
    val_pct = (val_count / total) * 100
    test_pct = (test_count / total) * 100
    print(f"Total de datos: {total}")
    print(f"Train: {train_count} ({train_pct}%)")
    print(f"Validation: {val_count} ({val_pct:.2f}%)")
    print(f"Test: {test_count} ({test_pct:.2f}%)")
    metadata.to_csv(f"/Users/oscar/Documents/Documents-Local/DeepLearning/Datasets/metadata_{int(train_pct)}.csv", index=False)


def do_split():
    args = argparse.Namespace(config="configs/exp01_config.json")
    config, _ = read_config(args)
    split_dataset(config)


def check_parameters(model):
    """
    Prints the total number of parameters, trainable parameters, and frozen parameters in the model.

    Args:
        model (torch.nn.Module): The PyTorch model to analyze.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params  # Non-trainable parameters

    print("INFO: Model Parameters Summary:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Frozen parameters: {frozen_params:,}")

if __name__ == "__main__":
    # do_split()
    check_devices()
    print("Done.")
    # copy_data(source_path = "/Users/oscar/Documents/Documents-Local/DeepLearning/Datasets/afhq", dest_folder = "/Users/oscar/Documents/Documents-Local/DeepLearning/Datasets/Animals_Face")
    # create_metadata("/Users/oscar/Documents/Documents-Local/DeepLearning/Datasets/Animal_Faces")
