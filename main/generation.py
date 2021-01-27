import torch
import sklearn
import sklearn.covariance
import numpy as np


def get_matrix_mean(model, loader, device, feature_list, num_classes):
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    model.eval()
    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)
    num_sample_per_class = np.empty(num_classes)
    num_sample_per_class.fill(0)
    num_output = len(feature_list)
    list_features = []

    for i in range(num_output):
        temp_list = []
        for j in range(num_classes):
            temp_list.append(0)
        list_features.append(temp_list)

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        out_features = model.feature_list(images)

        for i in range(num_output):
            out_features[i] = out_features[i].view(
                out_features[i].size(0), out_features[i].size(1), -1
            )
            out_features[i] = torch.mean(out_features[i].data, 2)

        for i in range(images.size(0)):  # batch size
            label = labels[i]
            if num_sample_per_class[label] == 0:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] = out[i].view(1, -1)
                    out_count += 1
            else:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] = torch.cat(
                        (list_features[out_count][label], out[i].view(1, -1)), 0
                    )
                    out_count += 1
            num_sample_per_class[label] += 1

    sample_class_mean = []
    out_count = 0
    for num_feature in feature_list:
        temp_list = torch.Tensor(num_classes, int(num_feature)).cuda()
        for j in range(num_classes):
            temp_list[j] = torch.mean(list_features[out_count][j], 0)
        sample_class_mean.append(temp_list)
        out_count += 1

    precision = []
    for k in range(num_output):
        X = 0
        for i in range(num_classes):
            if i == 0:
                X = list_features[k][i] - sample_class_mean[k][i]
            else:
                X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)
        # find inverse
        group_lasso.fit(X.cpu().numpy())
        temp_precision = group_lasso.precision_
        temp_precision = torch.from_numpy(temp_precision).float().cuda()
        print("temp_precision_size: ", temp_precision.size())
        precision.append(temp_precision)