import argparse
import tqdm
from torch.utils.data import DataLoader
import dataset
import utils


def get_data(config, classes):
    metadata = utils.read_metadata(config["data"]["metadata_file"])
    metadata.reset_index(drop=True, inplace=True)
    print(f"INFO: {metadata.shape=}")
    train_df = metadata[metadata["split"] == "train"].reset_index(drop=True)
    val_df = metadata[metadata["split"] == "val"].reset_index(drop=True)
    train = dataset.ImageDataset(config=config, metadata=train_df, classes=classes)
    val = dataset.ImageDataset(config=config, metadata=val_df, classes=classes)
    return train, val


def test():
    args = argparse.Namespace(config="configs/exp01_config.json")
    config, classes = utils.read_config(args)
    train, val = get_data(config, classes)
    for sample in train:
        print(
                f"img: {sample['img'].shape}, target: {sample['target']},"
                f"index: {sample['index']}"
            )
        break 
    for sample in val:
        print(
                f"img: {sample['img'].shape}, target: {sample['target']},"
                f"index: {sample['index']}"
            )
        break



if __name__ == "__main__":
    test()
    print("Done.")
