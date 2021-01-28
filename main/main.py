import argparse
import generation
import json
import os
import sys
import torch
import numpy as np

from distutils.util import strtobool
from glob import glob

sys.path.append("../preprocess/")
from preprocess import CowDataset, ImageTransform

sys.path.append("../models/")
from model import CustomDensenet


def make_dataloader(path, label, transforms):

    dataset = CowDataset(path, label, transform=transforms, phase="test", color=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batchsize, num_workers=1, shuffle=False
    )
    return dataloader


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("#device: ", device)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transforms = ImageTransform(mean, std)

    with open(args.train, "r") as f:
        db = json.load(f)

    train_path = db["train"]["path"]
    train_label = db["train"]["label"]
    train_dataloader = make_dataloader(train_path, train_label, transforms)

    model = CustomDensenet(num_classes=args.n_cls)
    model.to(device)
    model.load_state_dict(torch.load(args.weight))

    model.eval()
    torch.set_grad_enabled(False)
    temp_x = torch.rand(2, 3, 224, 224).cuda()

    temp_list = model.feature_list(temp_x)

    num_output = len(temp_list)
    feature_list = np.empty(num_output)
    count = 0
    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1

    # get covariance metrix and mean from training data
    cls_mean, precision = generation.get_matrix_mean(
        model, train_dataloader, device, feature_list, args.n_cls
    )

    # get mahalanobis scores of validation data
    for i in range(num_output):
        mahalanobis_score = generation.get_mahalanobis_score(
            model, val_dataloader, i, cls_mean, precision, args.n_cls, device
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str)
    parser.add_argument("--val", type=str)
    parser.add_argument("--n_cls", type=int, default=3)
    parser.add_argument("--weight", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument(
        "--concat", type=strtobool, help="whether concat garbage and negative or not"
    )
    parser.add_argument("--gpuid", type=str, default="0")
    parser.add_argument("--temperature", type=str)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
    main()
