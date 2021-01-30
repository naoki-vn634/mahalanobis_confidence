import argparse
import generation
import json
import os
import pickle
import sys
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
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

    train_path, train_label = db["train"]["path"], db["train"]["label"]
    train_dataloader = make_dataloader(train_path, train_label, transforms)

    val_path, val_label = [], []
    classes = ["no", "yes", "garbage"]
    class_dirs = glob(os.path.join(args.val, "*"))
    for class_dir in class_dirs:
        img_path = glob(os.path.join(class_dir, "*.jpg"))
        val_label.extend(
            [classes.index(os.path.basename(class_dir)) for _ in range(len(img_path))]
        )
        val_path.extend(img_path)

    val_dataloader = make_dataloader(val_path, val_label, transforms)

    model = CustomDensenet(num_classes=args.n_cls)
    model.to(device)
    model.load_state_dict(torch.load(args.weight))
    # print(model)

    model.eval()
    torch.set_grad_enabled(False)
    temp_x = torch.rand(2, 3, 224, 224).cuda()

    temp_list, layers_list, _ = model.feature_list(temp_x)

    num_output = len(temp_list)
    feature_list = np.empty(num_output)
    count = 0
    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1
    # get covariance metrix and mean from training data

    if args.matrix_mean:
        f0 = open(os.path.join(args.matrix_mean, "cls_mean.txt"), "rb")
        cls_mean = pickle.load(f0)
        f1 = open(os.path.join(args.matrix_mean, "precision.txt"), "rb")
        precision = pickle.load(f1)
    else:
        cls_mean, precision = generation.get_matrix_mean(
            model, train_dataloader, device, feature_list, args.n_cls
        )
        f_mean = open(os.path.join(args.output, "cls_mean.txt"), "wb")
        pickle.dump(cls_mean, f_mean)
        f_precision = open(os.path.join(args.output, "precision.txt"), "wb")
        pickle.dump(precision, f_precision)

    # get mahalanobis scores of validation data
    for i in range(num_output):
        num_features = layers_list[i]
        scores, labels, preds, preds_mah = generation.get_mahalanobis_score(
            model, val_dataloader, i, cls_mean, precision, args.n_cls, device
        )

        if args.concat:
            labels[np.where(labels == 2)[0]] = 0
            preds[np.where(preds == 2)[0]] = 0
            preds_mah[np.where(preds_mah == 2)[0]] = 0

        print(len(np.where(preds != preds_mah)[0]))

        confmat = confusion_matrix(labels, preds)
        print("mahalanobis_detector_matrix")
        print(confmat)
        count = 0
        wrong_score, correct_score = [], []
        for score, label, pred in zip(scores, labels, preds):
            if label != pred:
                wrong_score.append(score)
            else:
                correct_score.append(score)
        print(count)
        plt.figure()
        sns.distplot(wrong_score, label="wrong")
        sns.distplot(correct_score, label="correct")
        plt.legend()
        plt.xlabel("mahalanobis score")
        plt.savefig(
            os.path.join(
                args.output,
                f"feature_{num_features}*{num_features}_mahalanobis_score_mah.png",
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str)
    parser.add_argument("--val", type=str)
    parser.add_argument("--matrix_mean", type=str)
    parser.add_argument("--n_cls", type=int, default=3)
    parser.add_argument("--weight", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument(
        "--concat", type=strtobool, help="whether concat garbage and negative or not"
    )
    parser.add_argument("--gpuid", type=str, default="0")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
    main()
