"""
This module defines a custom dataset class for loading images and a test function
to verify its functionality.
"""

import os
import argparse
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import utils


class ImageDataset(Dataset):
    """Custom Dataset for loading images.

    Args:
        config (dict): Configuration dictionary.
        metadata (pd.DataFrame): Metadata containing image file names and labels.
        classes (dict): Dictionary mapping source labels to target labels.
        transform (bool, optional): Whether to apply transformations to the images.
        Defaults to True.
    """

    def __init__(self, config, metadata, classes, transform=True):
        super().__init__()
        self.config = config
        self.metadata = metadata
        self.classes = classes
        self.transform = transform
        self.show = False

    def __len__(self):
        """Returns the total number of samples."""
        return self.metadata.shape[0]

    def __getitem__(self, idx):
        """Generates one sample of data.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the image, target label, and index.
        """
        file = os.path.join(
            self.config["data"]["root_dir"], self.metadata.iloc[idx]["file_name"]
        )
        target = self.classes[self.metadata.iloc[idx]["source_label"]]
        img = utils.read_img(
            file, normalize=True, show=self.show, transform=self.transform
        )
        sample = {"img": img, "target": target, "index": idx}
        return sample


def test():
    """Function to test the ImageDataset class."""
    args = argparse.Namespace(config="configs/exp01_config.json")
    config, classes = utils.read_config(args)
    metadata = utils.read_metadata(config["data"]["metadata_file"])
    data = ImageDataset(config=config, metadata=metadata, classes=classes)
    test_loader = DataLoader(data, batch_size=4, shuffle=True)
    for i, batch in enumerate(
        tqdm(test_loader, desc="Processing data", total=len(test_loader))
    ):
        if i == 0:
            print(
                f"img: {batch['img'].shape}, target: {batch['target'].shape},"
                f"index: {batch['index']}"
            )


if __name__ == "__main__":
    test()
    print("Done.")
