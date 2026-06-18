import torch
from torchvision.io import decode_image
import os



class HandWrittenDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, label_maps,transform=None, target_transform = None):
        self.data_dir = data_dir
        self.label_maps = label_maps
        self.transform = transform
        self.target_transform = target_transform
        print( os.scandir(self.data_dir))
        self.data = list(map(lambda e:e.is_file(), os.scandir(self.data_dir)))
    

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, key):
        print("cc")

        el = (self.data or list(map(lambda e:e.is_file(), os.scandir(self.data_dir))))[key]
        print(el)
        # img = decode_image(el)
        # lbl = self.label_maps[(el).split('.')[0]]

        # if self.transform:
        #     img = self.transform(img)
        # if self.target_transform:
        #     lbl = self.target_transform(lbl)
        

        # return img, lbl

        
                
