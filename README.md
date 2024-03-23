# Computer Vision Datasets

[datasetninja.com](datasetninja.com) has an impressive collection of comptuer vision datasets. This package was built to let you easily download and use these computer vision datasets. 

## Installation
`pip install computer_vision_datasets` should work. If not, go to the repository and paste the file in your code.
The repository is in [https://github.com/franchesoni/computer_vision_datasets](https://github.com/franchesoni/computer_vision_datasets).


## Available datasets (code)
To check the available datasets, simply run
```python
from computer_vision_datasets import get_released_datasets
print(sorted(get_released_datasets().keys()))
```

## Download a dataset
Find the name of the dataset you want with the code above. Here we use 'ADE20K'.
```python
from computer_vision_datasets import download
download('ADE20K', '/your/destination/directory')
```

## The `SegDataset` class
There is a `SegDataset` class for segmentation datasets. You just need to point it to the dataset path. Example:

```python
ds = SegDataset(ds_path, split='test')
```
 
You can, for instance, wrap it in a Pytorch Dataset:

```python
from torch.utils.data import Dataset

class PyTorchWrapperDataset(Dataset):
    def __init__(self, ds):
        # super().__init__() is not needed since we're not overriding anything from the parent's __init__
        self.original_dataset = ds

    def __getitem__(self, index):
        # Assuming the original dataset uses __getitem__ to access items
        # If it uses a different method (like get_item), adjust accordingly.
        return self.original_dataset[index]

    def __len__(self):
        return len(self.original_dataset)
```








