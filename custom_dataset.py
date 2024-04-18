from torch.utils.data import Dataset
from PIL import Image
import torch

class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]
        image = Image.open(img_name)

        age = torch.tensor(self.dataframe.iloc[idx, 1], dtype=torch.float32)
        anatom_site = torch.tensor(self.dataframe.iloc[idx, 3], dtype=torch.float32)
        sex = torch.tensor(self.dataframe.iloc[idx, 4], dtype=torch.float32)
        label = torch.tensor(self.dataframe.iloc[idx, 5], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, age, anatom_site, sex, label