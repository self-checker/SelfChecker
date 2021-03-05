import json
import numpy as np

path_name = "cifar10/conv"
pred_labels = np.load("./tmp/" + path_name + "/pred_labels_test.npy")

num_classes = 10
num_layers = 9


# load indexes of selected layers per class
layers_agree, layers_accuracy, layers_weight, layers_accuracy_neg, layers_weight_neg = {}, {}, {}, {}, {}
for label in range(num_classes):
    filename = "./tmp/" + path_name + "/selected_layers_agree_" + str(label) + ".json"
    with open(filename, "r") as json_file:
        layers_agree_label = json.load(json_file)
    json_file.close()
    layers_agree.update(layers_agree_label)

    filename = "./tmp/" + path_name + "/selected_layers_accuracy_" + str(label) + ".json"
    with open(filename, "r") as json_file:
        layers_accuracy_label = json.load(json_file)
    json_file.close()
    layers_accuracy.update(layers_accuracy_label)

    filename = "./tmp/" + path_name + "/weights_" + str(label) + ".json"
    with open(filename, "r") as json_file:
        layers_weight_label = json.load(json_file)
    json_file.close()
    layers_weight.update(layers_weight_label)

    filename = "./tmp/" + path_name + "/selected_layers_accuracy_neg_" + str(label) + ".json"
    with open(filename, "r") as json_file:
        layers_accuracy_neg_label = json.load(json_file)
    json_file.close()
    layers_accuracy_neg.update(layers_accuracy_neg_label)

    filename = "./tmp/" + path_name + "/weights_neg_" + str(label) + ".json"
    with open(filename, "r") as json_file:
        layers_weight_neg_label = json.load(json_file)
    json_file.close()
    layers_weight_neg.update(layers_weight_neg_label)

# evaluate the performance of SelfChecker using the testing dataset
kde_acc_total = []
model_acc_total = []
TP = FP = FN = TN = TP_right = FP_right = TN_right = FN_right = 0
for label_agree in range(num_classes):
    print("\nlabel: {}".format(label_agree))
    # deal with instances per class according to their predictions
    pred_label_idx = np.where(pred_labels.T[-2] == label_agree)[0]

    # count the number of selected layers that agree with the final prediction
    num_selected_layers = len(layers_agree[str(label_agree)])
    pred_count = np.zeros([pred_labels[pred_label_idx].shape[0], num_selected_layers])
    misbh_count = np.zeros([pred_labels[pred_label_idx].shape[0], num_selected_layers])
    for idx in range(pred_labels[pred_label_idx].shape[0]):
        count_idx = 0
        for layer_idx in layers_agree[str(label_agree)]:
            if pred_labels[pred_label_idx[idx]][layer_idx] == pred_labels[pred_label_idx[idx]][-2]:
                pred_count[idx][count_idx] = 1
            if pred_labels[pred_label_idx[idx]][layer_idx] != pred_labels[pred_label_idx[idx]][-2]:
                misbh_count[idx][count_idx] = 1
            count_idx += 1
    sum_pred_example = np.sum(pred_count, axis=1)
    sum_misbh_example = np.sum(misbh_count, axis=1)

    # calculate confidence for positive cases
    pos_indexes = np.where(sum_misbh_example >= sum_pred_example)[0]

    KdePredPositive = sum_misbh_example >= sum_pred_example
    TrueMisBehaviour = pred_labels[pred_label_idx].T[-2] != pred_labels[pred_label_idx].T[-1]

    # calculate confusion metric
    TP += np.sum(TrueMisBehaviour & KdePredPositive)
    FP += np.sum(~TrueMisBehaviour & KdePredPositive)
    FN += np.sum(TrueMisBehaviour & ~KdePredPositive)
    # TN += np.sum(~TrueMisBehaviour & ~KdePredPositive)

    TP_idx = np.where(TrueMisBehaviour & KdePredPositive)[0]
    FP_idx = np.where(~TrueMisBehaviour & KdePredPositive)[0]
    TN_idx = np.where(~TrueMisBehaviour & ~KdePredPositive)[0]
    FN_idx = np.where(TrueMisBehaviour & ~KdePredPositive)[0]

    pred_of_label = np.zeros([pos_indexes.shape[0], num_classes])
    FP_idx_pos = []
    for label_acc in range(num_classes):
        if (str(label_agree) + str(label_acc)) not in layers_accuracy.keys():
            continue
        kde_preds = np.zeros([pos_indexes.shape[0], num_classes])
        count_idx = 0
        for idx in pos_indexes:
            for layer_idx in layers_accuracy[str(label_agree) + str(label_acc)]:
                kde_preds[count_idx][int(pred_labels[pred_label_idx[idx]][layer_idx])] += 1
            count_idx += 1
        kde_preds /= len(layers_accuracy[str(label_agree) + str(label_acc)])
        if layers_weight[str(label_agree) + str(label_acc)] == 0.0:
            kde_preds *= 1.0 / num_classes
        else:
            kde_preds *= layers_weight[str(label_agree) + str(label_acc)]
        pred_of_label.T[label_acc] = kde_preds.T[label_acc]

    kde_pred = np.argmax(pred_of_label, axis=1)

    idx_TP_idx = []
    idx_FP_idx = []
    for idx in TP_idx:
        idx_TP_idx.append(list(pos_indexes).index(idx))
    for idx in FP_idx:
        idx_FP_idx.append(list(pos_indexes).index(idx))
    TP_right += np.sum(kde_pred[idx_TP_idx] == pred_labels[pred_label_idx[TP_idx]].T[-1])
    FP_right += np.sum(kde_pred[idx_FP_idx] == pred_labels[pred_label_idx[FP_idx]].T[-1])
    FP -= np.sum(kde_pred[idx_FP_idx] == pred_labels[pred_label_idx[FP_idx]].T[-1])
    TN += np.sum(kde_pred[idx_FP_idx] == pred_labels[pred_label_idx[FP_idx]].T[-1])

    # calculate confidence for negative cases
    neg_indexes = np.where(sum_misbh_example < sum_pred_example)[0]
    pred_of_label_neg = np.zeros([neg_indexes.shape[0], num_classes])
    for label_acc in range(num_classes):
        if (str(label_agree) + str(label_acc)) not in layers_accuracy_neg.keys():
            continue
        kde_preds_neg = np.zeros([neg_indexes.shape[0], num_classes])
        count_idx = 0
        for idx in neg_indexes:
            for layer_idx in layers_accuracy_neg[str(label_agree) + str(label_acc)]:
                kde_preds_neg[count_idx][int(pred_labels[pred_label_idx[idx]][layer_idx])] += 1
            count_idx += 1
        kde_preds_neg /= len(layers_accuracy_neg[str(label_agree) + str(label_acc)])
        if layers_weight_neg[str(label_agree) + str(label_acc)] == 0.0:
            kde_preds_neg *= 1.0 / num_classes
        else:
            kde_preds_neg *= layers_weight_neg[str(label_agree) + str(label_acc)]
        pred_of_label_neg.T[label_acc] = kde_preds_neg.T[label_acc]

    kde_pred_neg = np.argmax(pred_of_label_neg, axis=1)

    idx_TN_idx = []
    idx_FN_idx = []
    for idx in TN_idx:
        idx_TN_idx.append(list(neg_indexes).index(idx))
    for idx in FN_idx:
        idx_FN_idx.append(list(neg_indexes).index(idx))
    TN_right += np.sum(kde_pred_neg[idx_TN_idx] == pred_labels[pred_label_idx[TN_idx]].T[-1])
    FN_right += np.sum(kde_pred_neg[idx_FN_idx] == pred_labels[pred_label_idx[FN_idx]].T[-1])
    TP += np.sum(kde_pred_neg[idx_FN_idx] == pred_labels[pred_label_idx[FN_idx]].T[-1])
    FP += np.sum(kde_pred_neg[idx_TN_idx] != pred_labels[pred_label_idx[TN_idx]].T[-1])
    TN += np.sum(kde_pred_neg[idx_TN_idx] == pred_labels[pred_label_idx[TN_idx]].T[-1])
    FN -= np.sum(kde_pred_neg[idx_FN_idx] == pred_labels[pred_label_idx[FN_idx]].T[-1])

TPR = TP / (TP + FN)
FPR = FP / (TN + FP)
F1 = 2 * TP / (2 * TP + FN + FP)
print("TP:{}, FP:{}, TN:{}, FN:{}, TPR:{:.6f}, FPR:{:.6f}, F1:{:.6f}".format(TP, FP, TN, FN, TPR, FPR, F1))
print("model accuracy: {}".format((FP + TN) / (TP + FP + TN + FN)))
print("kde accuracy: {}".format((TP_right + FP_right + FN_right + TN_right) / (TP + FP + TN + FN)))