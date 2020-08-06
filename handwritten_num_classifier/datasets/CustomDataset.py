
from PIL import Image
from pathlib import Path
import csv

import torch

# https://github.com/miyamotok0105/pytorch_handbook/blob/master/chapter4/section4_2.ipynb

class AdditionalDataset(torch.utils.data.Dataset):
  
    def __init__(self, dataset_dir_list, transform, dataset_name='additional'):
        # 指定する場合は前処理クラスを受け取ります。
        self.transform = transform
        self.dataset_name = dataset_name
        self.images = []
        self.labels = []
        # 1個のリストにします。
        
        for dataset_dir in dataset_dir_list:
            dataset_image_dir = dataset_dir / 'image'
            dataset_label = dataset_dir / 'csv/label.csv'
            
            with dataset_label.open('r') as f:
                reader = csv.reader(f)
                for file_, label in reader:
                    im_path = dataset_image_dir / file_
                    if not im_path.exists():
                        print(im_path, 'does not exist.')
                    self.images.append(str(im_path))
                    self.labels.append(int(label))

    def __getitem__(self, index):
        # インデックスを元に画像のファイルパスとラベルを取得します。
        image_path = self.images[index]
        label = self.labels[index]
        # 画像ファイルパスから画像を読み込みます。
        with open(image_path, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')
        # 前処理がある場合は前処理をいれます。
        if self.transform is not None:
            image = self.transform(image)
        # 画像とラベルのペアを返却します。
        return image, int(label), image_path, self.dataset_name
        
    def __len__(self):
        # ここにはデータ数を指定します。
        return len(self.images)