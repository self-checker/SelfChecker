import json
import numpy as np
import itertools
import sys

path_name = "cifar10/conv"
pred_labels = np.load("./tmp/" + path_name + "/pred_labels_valid.npy")

num_classes = 10
num_layers = 9
total_layers = [x for x in range(num_layers)]


def calculate_accuracy(layers, pred_label_idx):
    kde_preds = np.zeros([pred_label_idx.shape[0], num_classes])
    count_idx = 0
    for idx in pred_label_idx:
        for layer_idx in layers:
            kde_preds[count_idx][int(pred_labels[idx][layer_idx])] += 1
        count_idx += 1

    kde_pred = np.argmax(kde_preds, axis=1)
    kde_accuracy = np.mean(kde_pred == pred_labels[pred_label_idx].T[-1])

    return kde_accuracy


def selected_layer_condition(idx_in_origin):
    max_acc = 0
    selected_layers = None
    for count in range(1, num_layers + 1):
        for layers in itertools.combinations(total_layers, count):
            acc = calculate_accuracy(layers, idx_in_origin)

            if acc >= max_acc:
                max_acc = acc
                selected_layers = layers

    kde_acc = calculate_accuracy(selected_layers, idx_in_origin)
    model_acc = np.mean(pred_labels[idx_in_origin].T[-2] == pred_labels[idx_in_origin].T[-1])
    print("selected layers: {}, acc: {}".format(selected_layers, kde_acc))
    print("model acc: {}\n".format(model_acc))
    return selected_layers, kde_acc


def selected_layer_for_label(label, selected_layers_dict, weights_dict):
    # split dataset into subset according to their predictions
    pred_label_idx = np.where(pred_labels.T[-2] == label)[0]

    # count the number of layers that agree with the final prediction
    num_selected_layers = len(layers_agree[str(label)])
    pred_count = np.zeros([pred_labels[pred_label_idx].shape[0], num_selected_layers])
    misbh_count = np.zeros([pred_labels[pred_label_idx].shape[0], num_selected_layers])
    for idx in range(pred_labels[pred_label_idx].shape[0]):
        count_idx = 0
        for layer_idx in layers_agree[str(label)]:
            if pred_labels[pred_label_idx[idx]][layer_idx] == pred_labels[pred_label_idx[idx]][-2]:
                pred_count[idx][count_idx] = 1
            if pred_labels[pred_label_idx[idx]][layer_idx] != pred_labels[pred_label_idx[idx]][-2]:
                misbh_count[idx][count_idx] = 1
            count_idx += 1
    # calculate confidence
    sum_pred_example = np.sum(pred_count, axis=1)
    sum_misbh_example = np.sum(misbh_count, axis=1)
    pos_indexes = np.where(sum_misbh_example < sum_pred_example)[0]

    KdePredPositive = sum_misbh_example < sum_pred_example
    TrueMisBehaviour = pred_labels[pred_label_idx].T[-2] != pred_labels[pred_label_idx].T[-1]

    FP = np.sum(~TrueMisBehaviour & KdePredPositive)

    # searches for the best layer combination where the model predicts the input with label 'label_con' as 'label'
    for label_con in range(num_classes):
        pos_indexes_label = np.where(pred_labels[pred_label_idx[pos_indexes]].T[-1] == label_con)[0]
        print("label: {}, total_len: {}, label_con: {}, len: {}".format(label, pos_indexes.shape[0], label_con,
                                                                        pos_indexes_label.shape[0]))
        if pos_indexes_label.shape[0] == 0:
            print("check!")
            continue
        selected_layers_dict[str(label) + str(label_con)], kde_acc = selected_layer_condition(
            pred_label_idx[pos_indexes[pos_indexes_label]])
        if label_con == label:
            weights_dict[str(label) + str(label_con)] = pos_indexes_label.shape[0] * kde_acc / pos_indexes.shape[0]
        else:
            weights_dict[str(label) + str(label_con)] = pos_indexes_label.shape[0] * kde_acc / (
                        pos_indexes.shape[0] - FP)


selected_layers_dict = {}
weights_dict = {}

# # single-thread version
# for label in range(num_classes):
#     # load selected layers for alarm
#     filename = "./tmp/" + path_name + "/selected_layers_agree_" + str(label) + ".json"
#     with open(filename, "r") as json_file:
#         layers_agree = json.load(json_file)
#     json_file.close()
#
#     print("label: {}".format(label))
#     # generate selected layers per class
#     selected_layer_for_label(label, selected_layers_dict, weights_dict)
#     print("\n")
#
#     # save the index of selected layers per class
#     filename = "./tmp/" + path_name + "/selected_layers_accuracy_neg_" + str(label) + ".json"
#     with open(filename, 'w') as json_file:
#         json.dump(selected_layers_dict, json_file, ensure_ascii=False)
#     json_file.close()
#     filename = "./tmp/" + path_name + "/weights_neg_" + str(label) + ".json"
#     with open(filename, 'w') as json_file:
#         json.dump(weights_dict, json_file, ensure_ascii=False)
#     json_file.close()


# multi-thread version
label = int(sys.argv[1])
print("label: {}".format(label))

# load selected layers for alarm
filename = "./tmp/" + path_name + "/selected_layers_agree_" + str(label) + ".json"
with open(filename, "r") as json_file:
    layers_agree = json.load(json_file)
json_file.close()

# generate selected layers per class
selected_layer_for_label(label, selected_layers_dict, weights_dict)

# save the index of selected layers per class
filename = "./tmp/" + path_name + "/selected_layers_accuracy_neg_" + str(label) + ".json"
with open(filename, 'w') as json_file:
    json.dump(selected_layers_dict, json_file, ensure_ascii=False)
json_file.close()
filename = "./tmp/" + path_name + "/weights_neg_" + str(label) + ".json"
with open(filename, 'w') as json_file:
    json.dump(weights_dict, json_file, ensure_ascii=False)
json_file.close()