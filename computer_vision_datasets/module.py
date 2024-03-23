import requests
from pathlib import Path
import os
import json
import base64
import re

import zlib
import tarfile
import zipfile
import tqdm
import cv2
import numpy as np


class SegDataset:
    """
    A dataset class for semantic segmentation tasks.

    Args:
        ds_path (str): The path to the dataset directory.
        split (str, optional): The split of the dataset to load. Default is "train".

    Raises:
        ValueError: If no directory is found for the specified split.

    Attributes:
        img_list (list): A list of image file paths.
        ann_list (list): A list of annotation file paths.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns the image, masks, and class names for the given index.
        base64_2_data(s): Converts a base64 encoded string to a numpy array. Used for decoding masks.

    """

    def __init__(self, ds_path, split="train"):
        assert split.lower() in ("train", "test")

        # Helper function to find case-insensitive match for directory names
        def find_dir(path, dir_name):
            for name in os.listdir(path):
                if name.lower() == dir_name:
                    return name
            return None

        # try to load the split using different names, this allivates inconsistent dataset naming
        train_dir = (
            find_dir(ds_path, "train")
            or find_dir(ds_path, "labeled")
            or find_dir(ds_path, "train_data")
            or find_dir(ds_path, "train_val")
            or find_dir(ds_path, "training")
        )
        test_dir = (
            find_dir(ds_path, "test")
            or find_dir(ds_path, "test_data")
            or find_dir(ds_path, "test_a")
            or find_dir(ds_path, "testing")
            or find_dir(ds_path, "val")
            or find_dir(ds_path, "validation")
        )

        split_dir = test_dir if split.lower() == "test" else train_dir
        if split_dir is None:
            raise ValueError(f"No directory found for split '{split}'")
        self.img_list = [
            os.path.join(ds_path, split_dir, "img", fname)
            for fname in sorted(os.listdir(os.path.join(ds_path, split_dir, "img")))
        ]
        self.ann_list = [
            os.path.join(ds_path, split_dir, "ann", fname)
            for fname in sorted(os.listdir(os.path.join(ds_path, split_dir, "ann")))
        ]

        # check correctness
        assert (
            len(self.img_list) == len(self.ann_list)
            and Path(self.img_list[-1]).name.split(".")[0]
            == Path(self.ann_list[-1]).name.split(".")[0]
        ), "Number of images and annotations do not match"

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.img_list)

    def __getitem__(self, idx):
        """
        Returns the image, masks, and class names for the given index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the image, masks, and class names.
        """
        # load paths
        imgpath, annpath = self.img_list[idx], self.ann_list[idx]
        # get image
        img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # get annotation
        with open(annpath, "r") as f:
            ann = json.load(f)
        objects = ann["objects"]

        class_names = []
        masks = np.zeros((len(objects), img.height, img.width), dtype=np.uint8)
        for mind, obj in enumerate(objects):
            class_names.append(obj["classTitle"].lower().replace(" ", "-"))
            if "bitmap" in obj:
                seg = self.base64_2_data(obj["bitmap"]["data"])
                origin = obj["bitmap"]["origin"]
                masks[
                    mind,
                    origin[1] : origin[1] + seg.shape[0],
                    origin[0] : origin[0] + seg.shape[1],
                ] = (
                    seg > 0
                )
            elif "points" in obj:
                points = np.array(obj["points"]["exterior"])  # (col, row), 2
                H, W = ann["size"].values()
                masks[mind] = cv2.fillPoly(masks[mind], [points], 255) > 127

        return img, masks, class_names

    @staticmethod
    def base64_2_data(s: str) -> np.ndarray:
        """
        Convert base64 encoded string to numpy array.

        Args:
            s (str): The input base64 encoded string.

        Returns:
            np.ndarray: The bool numpy array.
        """
        z = zlib.decompress(base64.b64decode(s))
        n = np.frombuffer(z, np.uint8)

        imdecoded = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)
        if (len(imdecoded.shape) == 3) and (imdecoded.shape[2] >= 4):
            mask = imdecoded[:, :, 3].astype(bool)  # 4-channel imgs
        elif len(imdecoded.shape) == 2:
            mask = imdecoded.astype(bool)  # flat 2d mask
        else:
            raise RuntimeError("Wrong internal mask format.")
        return mask


def get_shortname1(name):
    """Convert a dataset name to a shortname. Separates TitleCase words with hyphens."""
    words = re.findall(r"[A-Z]+[a-z]*|[a-z]+|\d+", name.replace("_", " "))
    return "-".join(word.lower() for word in words)


def get_shortname2(name):
    """Convert a dataset name to a shortname. Separates snake_case words with hyphens."""
    return "-".join(word.lower() for word in name.replace("_", " ").split())


def convert_storage_to_mb(storage_str):
    """Convert a storage string to MB. The string is in the format 'number unit'."""
    num, unit = storage_str.split()
    num = float(num)
    if unit == "GB":
        return num * 1024
    elif unit == "MB":
        return num
    else:
        return None


def unpack_if_archive(path: str) -> str:
    """
    Unpacks an archive file to a directory of the same name.
    """
    if os.path.isdir(path):
        return path

    extraction_path = os.path.splitext(path)[0]

    if zipfile.is_zipfile(path):
        os.makedirs(extraction_path, exist_ok=True)

        with zipfile.ZipFile(path, "r") as zip_ref:
            total_files = len(zip_ref.infolist())

            with tqdm.tqdm(
                desc=f"Unpacking '{os.path.basename(path)}'",
                total=total_files,
                unit="file",
            ) as pbar:
                for file in zip_ref.infolist():
                    zip_ref.extract(file, extraction_path)
                    pbar.update(1)

            return extraction_path

    if tarfile.is_tarfile(path):
        os.makedirs(extraction_path, exist_ok=True)

        with tarfile.open(path, "r") as tar_ref:
            total_files = len(tar_ref.getnames())

            with tqdm.tqdm(
                desc=f"Unpacking '{os.path.basename(path)}'",
                total=total_files,
                unit="file",
            ) as pbar:
                for file in tar_ref.getnames():
                    tar_ref.extract(file, extraction_path)
                    pbar.update(1)

            return extraction_path

    return path


def download(
    dataset: str, dst_dir: str = "~/dataset-ninja/", unpack_archive: bool = True
) -> str:
    """
    Downloads a dataset from the specified source and saves it to the destination directory.

    Args:
        dataset (str): The name of the dataset to download.
        dst_dir (str, optional): The destination directory where the dataset will be saved. Defaults to "~/dataset-ninja/".
        unpack_archive (bool, optional): Whether to unpack the downloaded archive. Defaults to True.

    Returns:
        str: The path to the downloaded dataset.

    Raises:
        KeyError: If the specified dataset is not found in the available datasets.

    """
    dataset_name = dataset.lower().replace(" ", "-")

    dst_dir = os.path.expanduser(dst_dir)

    dst_path = os.path.join(dst_dir, f"{dataset_name}.tar")
    if os.path.exists(dst_path):
        print(f"Dataset '{dataset}' already downloaded")
        if unpack_archive and not os.path.isdir(os.path.splitext(dst_path)[0]):
            return unpack_if_archive(dst_path)
        return dst_path
    else:
        os.makedirs(dst_dir)

    data = get_released_datasets()
    try:
        data[dataset]
    except KeyError:
        raise KeyError(f"Dataset '{dataset}' not found. Please check dataset name")

    sly_url = data[dataset]["download_sly_url"]

    response = requests.get(sly_url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # Adjust the block size as needed

    with tqdm.tqdm(
        desc=f"Downloading '{dataset}'", total=total_size, unit="B", unit_scale=True
    ) as pbar:
        with open(dst_path, "wb") as file:
            for data in response.iter_content(block_size):
                file.write(data)
                pbar.update(len(data))

    if unpack_archive:
        return unpack_if_archive(dst_path)
    return dst_path


import requests


def get_released_datasets():
    """
    Retrieves a list of released datasets from a JSON file.
    The JSON file is the released_datasets.json file in the dataset-tools repository.

    Returns:
        list: A list of released datasets.
    """
    released_datasets_json_url = "https://raw.githubusercontent.com/supervisely/dataset-tools/main/dataset_tools/data/download_urls/released_datasets.json"
    response = requests.get(released_datasets_json_url)
    response.raise_for_status()
    released_datasets = response.json()
    return released_datasets
