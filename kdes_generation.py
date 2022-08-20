import os
from multiprocessing import Pool
import dill as pickle
import numpy as np
from scipy.stats import gaussian_kde
from tqdm import tqdm

from keras.models import Model

from utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # set GPU Limits


def _aggr_output(x):
    return [np.mean(x[..., j]) for j in range(x.shape[-1])]


def _get_saved_path(base_path, dtype, layer_names):
    """Determine saved path of ats and pred

    Args:
        base_path (str): Base save path.
        dtype (str): Name of dataset type (e.g., train, test, fgsm, ...).
        layer_names (list): List of layer names.

    Returns:
        ats_path: File path of ats.
        pred_path: File path of pred (independent of layers)
    """

    joined_layer_names = "_".join(layer_names[:5])
    return (
        os.path.join(
            base_path,
            dtype + "_" + joined_layer_names + "_ats" + ".npy",
        ),
        os.path.join(base_path, dtype + "_pred" + ".npy"),
    )


def get_ats(
        model,
        dataset,
        name,
        layer_names,
        save_path=None,
        batch_size=128,
        num_proc=10,
):
    """Extract activation traces of dataset from model.

    Args:
        model (keras model): Subject model.
        dataset (ndarray): Set of inputs fed into the model.
        name (str): Name of input set.
        layer_names (list): List of selected layer names.
        save_path (tuple): Paths of being saved ats and pred.
        batch_size (int): Size of batch when serving.
        num_proc (int): The number of processes for multiprocessing.

    Returns:
        ats (ndarray): Array of (layers, inputs, neuron outputs).
        pred (ndarray): Array of predicted classes.
    """

    outputs = [model.get_layer(layer_name).output for layer_name in layer_names]
    outputs.append(model.output)

    temp_model = Model(inputs=model.input, outputs=outputs)

    prefix = info("[" + name + "] ")
    p = Pool(num_proc)
    print(prefix + "Model serving")
    layer_outputs = temp_model.predict(dataset, batch_size=batch_size, verbose=1)
    pred_prob = layer_outputs[-1]
    pred = np.argmax(pred_prob, axis=1)
    layer_outputs = layer_outputs[:-1]

    print(prefix + "Processing ATs")
    ats = None
    for layer_name, layer_output in zip(layer_names, layer_outputs):
        print("Layer: " + layer_name)
        if layer_output[0].ndim == 3:
            # For convolutional layers
            layer_matrix = np.array(
                p.map(_aggr_output, [layer_output[i] for i in range(len(dataset))])
            )
        else:
            layer_matrix = np.array(layer_output)

        if ats is None:
            ats = layer_matrix
        else:
            ats = np.append(ats, layer_matrix, axis=1)
            layer_matrix = None

    if save_path is not None:
        np.save(save_path[0], ats)
        np.save(save_path[1], pred)

    return ats, pred


def _get_train_ats(model, x_train, x_valid, layer_names, args):
    """Extract ats of train and validation inputs. If there are saved files, then skip it.

    Args:
        model (keras model): Subject model.
        x_train (ndarray): Set of training inputs.
        x_valid (ndarray): Set of validation inputs.
        layer_names (list): List of selected layer names.
        args: keyboard args.

    Returns:
        train_ats (list): ats of train set.
        train_pred (list): pred of train set.
        valid_ats (list): ats of target set.
        valid_pred (list): pred of target set.
    """

    saved_train_path = _get_saved_path(args.save_path, "train", layer_names)
    if os.path.exists(saved_train_path[0]):
        print(infog("Found saved {} ATs, skip serving".format("train")))
        # In case train_ats is stored in a disk
        train_ats = np.load(saved_train_path[0])
        train_pred = np.load(saved_train_path[1])
    else:
        train_ats, train_pred = get_ats(
            model,
            x_train,
            "train",
            layer_names,
            save_path=saved_train_path,
        )
        print(infog("train ATs is saved at " + saved_train_path[0]))

    saved_valid_path = _get_saved_path(args.save_path, 'valid', layer_names)
    if os.path.exists(saved_valid_path[0]):
        print(infog("Found saved {} ATs, skip serving").format('valid'))
        # In case target_ats is stored in a disk
        valid_ats = np.load(saved_valid_path[0])
        valid_pred = np.load(saved_valid_path[1])
    else:
        valid_ats, valid_pred = get_ats(
            model,
            x_valid,
            "valid",
            layer_names,
            save_path=saved_valid_path,
        )
        print(infog("valid" + " ATs is saved at " + saved_valid_path[0]))

    return train_ats, train_pred, valid_ats, valid_pred


def _get_target_ats(model, x_test, layer_names, args):
    """Extract ats of train and validation inputs. If there are saved files, then skip it.

    Args:
        model (keras model): Subject model.
        x_test (ndarray): Set of testing inputs.
        layer_names (list): List of selected layer names.
        args: keyboard args.

    Returns:
        target_ats (list): ats of target set.
        target_pred (list): pred of target set.
    """

    saved_test_path = _get_saved_path(args.save_path, 'test', layer_names)
    if os.path.exists(saved_test_path[0]):
        print(infog("Found saved {} ATs, skip serving").format("test"))
        # In case target_ats is stored in a disk
        test_ats = np.load(saved_test_path[0])
        test_pred = np.load(saved_test_path[1])
    else:
        test_ats, test_pred = get_ats(
            model,
            x_test,
            "test",
            layer_names,
            save_path=saved_test_path,
        )
        print(infog("test" + " ATs is saved at " + saved_test_path[0]))

    return test_ats, test_pred


def _get_kdes(train_ats, class_matrix, args):
    """Kernel density estimation

    Args:
        train_ats (ndarray): List of activation traces in training set.
        class_matrix (dict): List of index of classes.
        args: Keyboard args.

    Returns:
        kdes (list): List of kdes per label if classification task.
        removed_cols (list): List of removed columns by variance threshold.
            To further reduce the computational cost, we ﬁlter out neurons
            whose activation values show variance lower than a pre-deﬁned threshold,
        max_kde (list): List of maximum kde values.
        min_kde (list): List of minimum kde values.
    """

    col_vectors = np.transpose(train_ats)
    variances = np.var(col_vectors, axis=1)
    removed_cols = np.where(variances < args.var_threshold)[0]

    kdes = {}
    max_kde = {}
    min_kde = {}
    tot = 0
    for label in tqdm(range(args.num_classes), desc="kde"):
        refined_ats = np.transpose(train_ats[class_matrix[label]])
        refined_ats = np.delete(refined_ats, removed_cols, axis=0)

        tot += refined_ats.shape[1]

        print("refined ats shape: {}".format(refined_ats.shape))

        if refined_ats.shape[0] == 0:
            print(
                warn("all ats were removed by threshold {}".format(args.var_threshold))
            )
            break

        print("refined ats min max {} ; {} ".format(refined_ats.min(), refined_ats.max()))

        kdes[label] = gaussian_kde(refined_ats)
        outputs = kdes[label](refined_ats)
        max_kde[label] = np.max(outputs)
        min_kde[label] = np.min(outputs)
        print("min_kde: %s" % min_kde[label])
        print("max_kde: %s" % max_kde[label])

    print("gaussian_kde(refined_ats) shape[1] sum: {}".format(tot))

    print(infog("The number of removed columns: {}".format(len(removed_cols))))

    return kdes, removed_cols, max_kde, min_kde


def cal_print_f1(TP, FP, FN, TN):
    TPR = TP / (TP + FN)
    FPR = FP / (TN + FP)

    f1_score = 2 * TP / (2 * TP + FN + FP)

    print(info("TP: {}  FN: {}  FP: {}  TN: {}".format(TP, FN, FP, TN)))
    print(infog("TPR: {}  FPR: {}  F-1: {}".format(TPR, FPR, f1_score)))

    return TPR, FPR, f1_score


def kde_values_analysis(kdes, removed_cols, target_ats, target_label, target_pred, target_name, args):
    kde_values = np.zeros([target_ats.shape[0], args.num_classes])
    # obtain 10 kde values for each test
    for label in tqdm(range(len(kdes)), target_name):
        refined_ats = np.transpose(target_ats)
        refined_ats = np.delete(refined_ats, removed_cols, axis=0)
        kde_values.T[label] = kdes[label](refined_ats)

    pred_labels = np.argmax(kde_values, axis=1)
    print("model accuracy: {}, {}".format(target_name, np.mean(np.array(target_pred) == np.array(target_label))))
    print("kde accuracy:{}, {}".format(target_name, np.mean(np.array(pred_labels) == np.array(target_label))))

    KdePredPositive = pred_labels != target_pred
    TrueMisBehaviour = target_label != target_pred

    TP = np.sum(TrueMisBehaviour & KdePredPositive)
    FP = np.sum(~TrueMisBehaviour & KdePredPositive)
    TN = np.sum(~TrueMisBehaviour & ~KdePredPositive)
    FN = np.sum(TrueMisBehaviour & ~KdePredPositive)

    _, _, _ = cal_print_f1(TP, FP, FN, TN)

    return pred_labels


def _get_model_output_idx(model, layer_names):
    # return param
    output_idx_map = {}

    # local tmp param
    start = 0
    end = 0
    layer_idx_map = {}

    # mapping layer names to layer
    for layer in model.layers:
        if layer.name in layer_names:
            layer_idx_map[layer.name] = layer

    assert len(layer_names) == len(layer_idx_map)

    # calc each layer output idx
    for layer_name in layer_names:
        layer = layer_idx_map[layer_name]
        name = layer.name
        output_shape = layer.output_shape
        end += output_shape[-1]
        output_idx_map[name] = (start, end)

        start = end

    return output_idx_map


def save_results(fileName, obj):
    dir = os.path.dirname(fileName)
    if not os.path.exists(dir):
        os.makedirs(dir)

    f = open(fileName, 'wb')
    pickle.dump(obj, f)


def train_fetch_kdes(model, x_train, x_valid, y_train, y_valid, layer_names, args):
    """kde functions and kde inferred classes per class for all layers

    Args:
        model (keras model): Subject model.
        x_train (ndarray): Set of training inputs.
        x_valid (ndarray): Set of validation inputs.
        y_train (ndarray): Ground truth of training inputs.
        y_valid (ndarray): Ground truth of validation inputs.
        layer_names (list): List of selected layer names.
        args: Keyboard args.

    Returns:
        None
        There is no returns but will save kde functions per class and inferred classes for all layers
    """
    print(info("### y_train len:{} ###".format(len(y_train))))
    print(infog("### y_valid len:{} ###".format(len(y_valid))))

    # obtain the number of neurons for each layer
    model_output_idx = _get_model_output_idx(model, layer_names)

    # generate feature vectors for each layer on training, validation set
    all_train_ats, train_pred, all_valid_ats, valid_pred = _get_train_ats(model, x_train, x_valid, layer_names, args)

    # obtain the input indexes for each class
    class_matrix = {}
    for i, label in enumerate(np.reshape(y_train, [-1])):
        if label not in class_matrix:
            class_matrix[label] = []
        class_matrix[label].append(i)

    pred_labels_valid = np.zeros([x_valid.shape[0], (len(layer_names) + 1)])
    layer_idx = 0
    for layer_name in layer_names:
        print(info("Layer: {}".format(layer_name)))
        idx = int(layer_name.split("_")[1])
        # in case the var_threshold is unsuitable
        if idx == 8:
            kdes_file = args.save_path + "/kdes-pack/%s" % layer_name
            if os.path.exists(kdes_file):
                print("remove existing kde functions!")
                os.remove(kdes_file)
            args.var_threshold = 2e-1
        if idx == 9:
            kdes_file = args.save_path + "/kdes-pack/%s" % layer_name
            if os.path.exists(kdes_file):
                print("remove existing kde functions!")
                os.remove(kdes_file)
            args.var_threshold = 0
        if layer_name == "dense_1":
            kdes_file = args.save_path + "/kdes-pack/%s" % layer_name
            if os.path.exists(kdes_file):
                print("remove existing kde functions!")
                os.remove(kdes_file)
            args.var_threshold = 0
        print("layer_index: {}, var_threshold: {}".format(idx, args.var_threshold))

        prefix = info("[" + layer_name + "] ")

        # get layer names ats
        (start_idx, end_idx) = model_output_idx[layer_name]
        train_ats = all_train_ats[:, start_idx:end_idx]
        valid_ats = all_valid_ats[:, start_idx:end_idx]

        # generate kde functions per class and layer
        kdes_file = args.save_path + "/kdes-pack/%s" % layer_name
        if os.path.exists(kdes_file):
            file = open(kdes_file, 'rb')
            (kdes, removed_cols, max_kde, min_kde) = pickle.load(file)
            print(infog("The number of removed columns: {}".format(len(removed_cols))))
            print(info("load kdes from file:" + kdes_file))
        else:
            print(info("calc kdes..."))
            kdes, removed_cols, max_kde, min_kde = _get_kdes(train_ats, class_matrix, args)
            save_results(args.save_path + "/kdes-pack/%s" % layer_name, (kdes, removed_cols, max_kde, min_kde))

        # generate inferred classes for each layer
        print(prefix + "Fetching KDE inference")
        pred_labels = kde_values_analysis(kdes, removed_cols, valid_ats, y_valid, valid_pred, "valid", args)
        pred_labels_valid.T[layer_idx] = pred_labels

        layer_idx += 1

    # save all inferred classes for evaluation
    pred_labels_valid.T[-1] = valid_pred
    pred_labels_concat_valid = np.concatenate((pred_labels_valid, np.reshape(y_valid, [y_valid.shape[0], 1])), axis=1)
    np.save(args.save_path + "/pred_labels_valid", pred_labels_concat_valid)


def test_fetch_kdes(model, x_test, y_test, layer_names, args):
    """kde functions and kde inferred classes per class for all layers

    Args:
        model (keras model): Subject model.
        x_test (ndarray): Set of testing inputs.
        y_test (ndarray): Ground truth of testing inputs.
        layer_names (list): List of selected layer names.
        args: Keyboard args.

    Returns:
        None
        There is no returns but will save kde functions per class and inferred classes for all layers
    """

    print(infog("### y_test len:{} ###".format(len(y_test))))

    # obtain the number of neurons for each layer
    model_output_idx = _get_model_output_idx(model, layer_names)

    # generate feature vectors for each layer on training, validation set
    all_test_ats, test_pred = _get_target_ats(model, x_test, layer_names, args)

    pred_labels_test = np.zeros([x_test.shape[0], (len(layer_names) + 1)])
    layer_idx = 0
    for layer_name in layer_names:
        print(info("Layer: {}".format(layer_name)))

        prefix = info("[" + layer_name + "] ")

        # get layer names ats
        (start_idx, end_idx) = model_output_idx[layer_name]
        test_ats = all_test_ats[:, start_idx:end_idx]

        # generate kde functions per class and layer
        kdes_file = args.save_path + "/kdes-pack/%s" % layer_name
        file = open(kdes_file, 'rb')
        (kdes, removed_cols, max_kde, min_kde) = pickle.load(file)
        print(infog("The number of removed columns: {}".format(len(removed_cols))))
        print(info("load kdes from file:" + kdes_file))

        # generate inferred classes for each layer
        print(prefix + "Fetching KDE inference")
        pred_labels = kde_values_analysis(kdes, removed_cols, test_ats, y_test, test_pred, "test", args)
        pred_labels_test.T[layer_idx] = pred_labels

        layer_idx += 1

    # save all inferred classes for evaluation
    pred_labels_test.T[-1] = test_pred
    pred_labels_concat_test = np.concatenate((pred_labels_test, np.reshape(y_test, [y_test.shape[0], 1])), axis=1)
    np.save(args.save_path + "/pred_labels_test", pred_labels_concat_test)