import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils import *
from torchvision import transforms
from torch.nn.functional import one_hot
import pandas as pd
from torchvision.transforms import functional as F
# from torchvision.transforms.functional import to_pil_image
import cv2
import random
from scipy.ndimage import zoom
from scipy import ndimage
from torchvision import transforms as T
from transformers import BertTokenizer
torch.set_printoptions(profile="full")  # set print the whole tensor
np.set_printoptions(threshold=np.inf)
from utils import rand_bbox


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def random_zoom(image, label):
    # mode = ["BICUBIC", "NEAREST"]
    min_zoom = 0.95
    max_zoom = 1.2
    zoom_factor = (min_zoom, max_zoom)
    image_pil = Image.fromarray(np.uint8(image))
    label_pil = Image.fromarray(np.uint8(label))
    image_pil_RandomAffine = transforms.RandomAffine(degrees=0, translate=(0, 0), scale=zoom_factor, shear=0)
    label_pil_RandomAffine = transforms.RandomAffine(degrees=0, translate=(0, 0), scale=zoom_factor, shear=0)
    image = np.array(image_pil_RandomAffine(image_pil))
    label = np.array(label_pil_RandomAffine(label_pil))
    return image, label

def correct_dims(*images):
    corr_images = []
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)

    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images


def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()


class MyDataset(Dataset):
    """QaTa-COV19-v2 image, text-embedding, and mask dataset."""
    def __init__(self, path, mode):
        self.path = path
        self.mode = mode
        self.name = os.listdir(os.path.join(path, 'Ground-truths'))
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        name = self.name[index]

        # if name == "mask_sub-S08077_ses-E18680_run-1_bp-chest_vp-ap_cr.png":
        #     print("1 index {} ".format(index))
        # if name == "mask_sub-S09458_ses-E16368_run-1_bp-chest_vp-ap_dx.png":
        #     print("2 index {} ".format(index))

        # image
        image_path = os.path.join(self.path, 'Images', name[5:])
        image = cv2.imread(image_path)
        # image = cv2.resize(image, (224, 224))


        # annotation
        # annotation = np.squeeze(np.load("temp_file/emptynpy.npy"), axis=0)


        annotation_path = os.path.join(self.path, 'Text_Emb', name.replace("png", "npy"))
        annotation = np.squeeze(np.load(annotation_path), axis=0)

        # if random.random() > 0.5:
        # #     # annotation = None
        #     annotation = np.squeeze(np.load("temp_file/emptynpy.npy"), axis=0)

        # mask
        mask_path = os.path.join(self.path, 'Ground-truths', name)
        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (224, 224))
        mask[mask <= 0] = 0
        mask[mask > 0] = 1

        image, mask = correct_dims(image, mask)
        # print(np.amax(mask))
        # mask = torch.zeros((False, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)
        '''
        # ------------------------------  CutMix  ------------------------------------------
        prob = 30  # Set to 0 to disable CutMix.
        if random.randint(0, 99) < prob and self.mode == "train":
            rand_index = random.randint(0, 5716)
            # print(rand_index)
            rand_name = self.name[rand_index]
            rand_image_path = os.path.join(self.path, 'Images', rand_name[5:])

            rand_image = cv2.imread(rand_image_path)
            rand_image = cv2.cvtColor(rand_image, cv2.COLOR_BGR2RGB)
            rand_image = cv2.resize(rand_image, (224, 224), interpolation=cv2.INTER_LINEAR)

            rand_masks_path = os.path.join(self.path, 'Ground-truths', rand_name)
            rand_masks = cv2.imread(rand_masks_path, cv2.IMREAD_GRAYSCALE) / 255
            rand_masks = cv2.resize(rand_masks, (224, 224), interpolation=cv2.INTER_LINEAR)

            rand_image, rand_masks = correct_dims(rand_image, rand_masks)

            lam = np.random.beta(1, 1)
            bbx1, bby1, bbx2, bby2 = rand_bbox(image.shape, lam)
            # print(image.shape, rand_image.shape, mask.shape, rand_masks.shape) (224, 224, 3) (224, 224, 3) (224, 224, 1) (224, 224, 1)

            image[bbx1:bbx2, bby1:bby2, :] = rand_image[bbx1:bbx2, bby1:bby2, :]
            mask[bbx1:bbx2, bby1:bby2] = rand_masks[bbx1:bbx2, bby1:bby2]
            # plt.imshow(image)
            # plt.axis('off')
            # plt.show()
            # plt.imshow(mask)
            # plt.axis('off')
            # plt.show()
        # ---------------------------------  CutMix  ---------------------------------------
        '''
        image, mask = image.astype(np.uint8), mask.astype(np.uint8)
        image, mask = F.to_pil_image(image), F.to_pil_image(mask)
        output_size = [224, 224]
        x, y = image.size

        if self.mode == "train":

            if random.random() > 0.8:
                image, mask = random_rotate(image, mask)
            elif random.random() > 0.8:
                image, mask = random_zoom(image, mask)
            # elif random.random() > 0.7:
            #     image, mask = random_rot_flip(image, mask)


        if x != output_size[0] or y != output_size[1]:
            image = zoom(image, (output_size[0] / x, output_size[1] / y), order=3)  # why not 3?
            mask = zoom(mask, (output_size[0] / x, output_size[1] / y), order=0)
        image = F.to_tensor(image)
        mask = to_long_tensor(mask).squeeze()


        # print(annotation.shape)
        return image, annotation, mask


class MyDataset_ema(Dataset):
    """Unlabeled QaTa-COV19-v2 data for the EMA branch."""
    def __init__(self, path, mode):
        self.path = path
        self.mode = mode
        self.name = os.listdir(os.path.join(path, 'Images'))
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        name = self.name[index]
        # image
        image_path = os.path.join(self.path, 'Images', name)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))

        # annotation
        annotation = np.zeros((10, 256), dtype=np.float32)

        # annotation_path = os.path.join(self.path, 'Text_Emb', name.replace("png", "npy"))
        # annotation = np.squeeze(np.load(annotation_path), axis=0)
        image = correct_dims(image)
        image= image.astype(np.uint8)
        image= F.to_pil_image(image)
        image = F.to_tensor(image)
        return image, annotation


if __name__ == '__main__':
    # QaTa-COV19-v2: Train 5716, Val 1429, Test 2113
    train_data_path = r'QaTa-COV19-v2/Train_Set'
    val_data_path = r'QaTa-COV19-v2/Val_Set'
    test_data_path = r'QaTa-COV19-v2/Test_Set'

    # MoNuSeg
    # train_data_path = r'datasets/MoNuSeg/Train_Folder'
    # val_data_path = r'datasets/MoNuSeg/Val_Folder'
    # test_data_path = r'datasets/MoNuSeg/Test_Folder'

    # data = MyDataset(val_data_path, mode="test")[0]
    # print(data[0].shape, data[1].shape, data[2].shape)

    data = MyDataset(train_data_path, mode="train")[0]
    print(data[0].shape, data[1].shape, data[2].shape)


    # Loader
    train_loader = DataLoader(MyDataset(train_data_path, mode="train"),
                              batch_size=1,
                              shuffle=True)
    val_loader = DataLoader(MyDataset(val_data_path, mode="test"),
                              batch_size=1,
                              shuffle=True)
    test_loader = DataLoader(MyDataset(test_data_path, mode="test"),
                             batch_size=1,
                             shuffle=False)

    train_dataset_length, val_dataset_length = len(train_loader), len(val_loader)
    print("train dataset length: ", train_dataset_length, "\nval dataset length: ", val_dataset_length)
    test_dataset_length = len(test_loader)
    print("test dataset length:", test_dataset_length)

    # ========================================
    ema_data_path = r'QaTa-COV19-v2/Aug'
    ema_loader = DataLoader(MyDataset_ema(ema_data_path, mode="ema"),
                              batch_size=1,
                              shuffle=True)
