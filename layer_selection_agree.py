import json
import numpy as np
import itertools
import sys

path_name = "cifar10/conv"
pred_labels = np.load("./tmp/" + path_name + "/pred_labels_valid.npy")

num_classes = 10
num_layers = 9
total_layers = [x for x in range(num_layers)]


def calculate_F1(layers, pred_label_idx):
    # count the number of layers that agree with the final prediction
    num_selected_layers = len(layers)
    pred_count = np.zeros([pred_labels[pred_label_idx].shape[0], num_selected_layers])
    misbh_count = np.zeros([pred_labels[pred_label_idx].shape[0], num_selected_layers])
    for idx in range(pred_labels[pred_label_idx].shape[0]):
        count_idx = 0
        for layer_idx in layers:
            if pred_labels[pred_label_idx[idx]][layer_idx] == pred_labels[pred_label_idx[idx]][-2]:
                pred_count[idx][count_idx] = 1
            if pred_labels[pred_label_idx[idx]][layer_idx] != pred_labels[pred_label_idx[idx]][-2]:
                misbh_count[idx][count_idx] = 1
            count_idx += 1
    # calculate confidence
    sum_pred_example = np.sum(pred_count, axis=1)
    sum_misbh_example = np.sum(misbh_count, axis=1)
    KdePredPositive = sum_misbh_example >= sum_pred_example

    TrueMisBehaviour = pred_labels[pred_label_idx].T[-2] != pred_labels[pred_label_idx].T[-1]

    # calculate confusion metric
    TP = np.sum(TrueMisBehaviour & KdePredPositive)
    FP = np.sum(~TrueMisBehaviour & KdePredPositive)
    TN = np.sum(~TrueMisBehaviour & ~KdePredPositive)
    FN = np.sum(TrueMisBehaviour & ~KdePredPositive)

    TPR = TP / (TP + FN)
    FPR = FP / (TN + FP)
    F1 = 2 * TP / (2 * TP + FN + FP)

    return TPR, FPR, F1, TP, FP, FN, TN


def selected_layer_for_label(label, selected_layers_dict):
    max_f1 = 0
    selected_layers = None
    # split dataset into subset according to their predictions
    pred_label_idx = np.where(pred_labels.T[-2] == label)[0]
    for count in range(1, num_layers+1):
        print("count: {}".format(count))
        for layers in itertools.combinations(total_layers, count):
            TPR, FPR, F1, TP, FP, FN, TN = calculate_F1(layers, pred_label_idx)
            if F1 >= max_f1:
                max_f1 = F1
                selected_layers = layers

    selected_layers_dict[str(label)] = selected_layers
    print("selected layers: {}".format(selected_layers))
    TPR, FPR, F1, TP, FP, FN, TN = calculate_F1(selected_layers, pred_label_idx)
    print("TPR:{:.6f} FPR:{:.6f} F1:{:.6f} TP:{} FP:{} FN:{} TN:{}".format(TPR, FPR, F1, TP, FP, FN, TN))


selected_layers_dict = {}

# # single-thread version
# for label in range(num_classes):
#     print("label: {}".format(label))
#     # generate selected layers per class
#     selected_layer_for_label(label, selected_layers_dict)
#     print("\r\n")
#
#     # save the index of selected layers per class
#     filename = "./tmp/" + path_name + "/selected_layers_agree_" + str(label) + ".json"
#     with open(filename, 'w') as json_file:
#         json.dump(selected_layers_dict, json_file, ensure_ascii=False)
#     json_file.close()


# multi-thread version
label = int(sys.argv[1])
print("label: {}".format(label))
# generate selected layers per class
selected_layer_for_label(label, selected_layers_dict)

# save the index of selected layers per class
filename = "./tmp/" + path_name + "/selected_layers_agree_" + str(label) + ".json"
with open(filename, 'w') as json_file:
    json.dump(selected_layers_dict, json_file, ensure_ascii=False)
json_file.close()
