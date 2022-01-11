import cv2
import torch
import torch.utils.data


class PNGDataset(torch.utils.data.Dataset):
    def __init__(self, imgs_df, img_ext, transform=None):
        self.imgs_df = imgs_df
        self.img_ext = img_ext
        self.transform = transform

    def __len__(self):
        return len(self.imgs_df)

    def __getitem__(self, idx):
        pic_meta = self.imgs_df.iloc[idx]
        img_id = pic_meta['id']
        img = cv2.imread(pic_meta['path'])
        class_index = pic_meta['class']
        class_index = torch.tensor(int(class_index))
        if self.transform is not None:
            augmented = self.transform(image=img)
            img = augmented['image']
        return img, class_index, img_id
