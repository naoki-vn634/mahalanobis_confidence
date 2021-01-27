import cv2
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image


class ImageTransform(object):
    def __init__(self, mean, std, resize=224):
        self.transform = {
            "train": transforms.Compose(
                [
                    # transforms.Resize(resize),
                    # transforms.RandomApply(
                    #     [transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)],
                    #     p=0.8,
                    # ),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ),
            "test": transforms.Compose(
                [
                    transforms.ToTensor(),
                    # transforms.Resize(resize),
                    transforms.Normalize(mean, std),
                ]
            ),
        }

    def __call__(self, image, phase):
        img_transformed = self.transform[phase](image)
        return img_transformed


class EncoderInputTransform(object):
    def __init__(self):
        self.transform = {
            "train": transforms.Compose(
                [
                    # transforms.Resize(resize),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean,std),
                ]
            ),
            "test": transforms.Compose(
                [
                    # transforms.Resize(resize),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean,std),
                ]
            ),
        }

    def __call__(self, image, phase):
        img_transformed = self.transform[phase](image)
        return img_transformed


class CowDatasetWithPath(object):
    def __init__(self, file_list, label_list, transform, phase):
        self.file_list = file_list
        self.label_list = label_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        image = self.file_list[index]
        img = cv2.imread(image)
        (h, w) = img.shape[:2]
        if not (h == 224 and w == 224):
            img = cv2.resize(img, (224, 224))
        img_transformed = self.transform(img, self.phase)

        return img_transformed, int(self.label_list[index]), self.file_list[index]


class CowDataset(object):
    def __init__(self, file_list, label_list, transform, phase, color=False):
        self.file_list = file_list
        self.label_list = label_list
        self.transform = transform
        self.phase = phase
        self.color = color
        self.wrong = torch.zeros(len(file_list))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        image = self.file_list[index]
        if self.color:
            img = Image.open(image)
            h, w = img.size
            if not (h == 224 and w == 224):
                img = img.resize((224, 224))

        else:
            img = cv2.imread(image)
            (h, w) = img.shape[:2]

            if not (h == 224 and w == 224):
                img = cv2.resize(img, (224, 224))
        img_transformed = self.transform(img, self.phase)

        return img_transformed, int(self.label_list[index])
