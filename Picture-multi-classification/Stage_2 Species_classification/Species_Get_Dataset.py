import os
import pandas as pd
from PIL import Image


class MyDataset:
    # define __init__
    def __init__(self, file_dir, anno_file, transform=None):
        self.file_dir = file_dir
        self.anno_file = anno_file
        self.transform = transform
        if not os.path.isfile(self.anno_file):
            print(self.anno_file + 'does not exist!')
        self.file_info = pd.read_csv(anno_file, index_col=0)
        self.size = len(self.file_info)
    # define __len__
    def __len__(self):
        return self.size
    # define __getitem__
    def __getitem__(self, idx):
        # get image
        path = self.file_info['path'][idx]
        if not os.path.isfile(path):
            print(path + ' does not exist!')
            return None
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # get annotation
        species = int(self.file_info['species'][idx])
        # make sample as dictionary and return sample
        sample = {'image': image, 'species': species}
        return sample
