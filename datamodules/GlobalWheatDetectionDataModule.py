import ast
import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.utils.data as data
from skimage import io
import torch.nn.functional as F

'''
    The dataset is made of a dataframe with
    
    image id, width, height, bbox 
    
    returns:
                    IMAGE                                   IMAGES
        (batch_size, width, height, channels), (batch_size, numBbox, bboxcoordinates)
        
    Note that to stack multiple images' bboxes the number should be equals.
    for this reason we should set the maximum number of bboxes that could be present in an image in such a way 
    we wont find any problem increasing the batch size,
    otherwise is possible to set batch size = 1 in a Dataloader avoiding padding
'''

'''
    Note:
    https://www.kaggle.com/competitions/global-wheat-detection/data
    
    The GlobalWheatDataset has the bounding boxes in the format:
        [xmin, ymin, width, height]
'''

class GlobalWheatDetectionDataset(data.Dataset):


    def __init__(self, csv_file, image_root_dir, img_extension = ".jpg", transform= None, pad=None):
        super().__init__()
        self.img_extension = img_extension
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = image_root_dir
        self.transform = transform
        self.pad = pad

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        '''
            The image id is the first column
        '''
        img_id = self.landmarks_frame.iloc[idx, 0]
        img_name = os.path.join(self.root_dir,
                                img_id)
        '''
            Read the image
        '''
        image = io.imread(img_name + self.img_extension)
        '''
            Get Boxes
        '''
        bbox = self.landmarks_frame[self.landmarks_frame.image_id == img_id].bbox.values ## string of float list
        bboxex = [ast.literal_eval(x) for x in bbox]
        bbox_tensor = torch.tensor(bboxex)
        if self.pad:
            bbox_tensor = F.pad(input=bbox_tensor, pad=(0, 0, self.pad - bbox_tensor.shape[-2], 0), mode='constant', value=0)


        '''
            Apply image transformation
        '''
        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'bboxes': bbox_tensor}
        return sample




class GlobalWheatDetectionDataModule(pl.LightningDataModule):

    def __init__(self, data_path):
        self.data_path = data_path





    def train_dataloader(self):
        return

    def val_dataloader(self):
        return









def main():

    dataset = GlobalWheatDetectionDataset(csv_file="../../Datasets/global-wheat-detection/global-wheat-detection/train.csv",
                                          image_root_dir="../../Datasets/global-wheat-detection/global-wheat-detection/train",
                                          pad=100)

    
    out = dataset.__getitem__(1)

    print(out['image'].shape, out['bboxes'].shape)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=100)

    it = iter(dataloader)
    for _ in range(2):
        sample = next(it)
        print(sample['image'].shape, sample['bboxes'].shape)


if __name__ == "__main__":
    main()