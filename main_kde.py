# %%
import argparse
import os

from keras.datasets import mnist, fashion_mnist, cifar10, cifar100
from keras.models import load_model

from kdes_generation import fetch_kdes
from utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # set GPU Limits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="cifar10")
    parser.add_argument("--m", "-m", help="Model", type=str, default="conv")
    parser.add_argument("--save_path", "-save_path", help="Save path", type=str, default="./tmp/")
    parser.add_argument("--batch_size", "-batch_size", help="Batch size", type=int, default=128)
    parser.add_argument("--var_threshold", "-var_threshold", help="Variance threshold", type=float, default=1e-5)
    parser.add_argument("--num_classes", "-num_classes", help="The number of classes", type=int, default=10)

    args = parser.parse_args()
    args.save_path = args.save_path + args.d + "/" + args.m + "/"
    dir = os.path.dirname(args.save_path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    print(args)

    # layer names
    if args.m == "conv":
        layer_names = []
        for i in range(1, 10):
            layer_names.append("activation_" + str(i))
    elif args.m == "vgg16":
        layer_names = []
        for i in range(1, 16):
            layer_names.append("activation_" + str(i))
    else:
        layer_names = []
        for i in range(1, 20):
            layer_names.append("activation_" + str(i))
        layer_names.append("dense_1")

    # load dataset and models
    x_train_total = x_test = y_train_total = y_test = model = None
    if args.d == "mnist":
        (x_train_total, y_train_total), (x_test, y_test) = mnist.load_data()
        x_train_total = x_train_total.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        num_train = 50000

        print(infog("y_train len:{}".format(len(y_train_total))))
        print(infog("y_test len:{}".format(len(y_test))))
        # Load pre-trained model.
        model = load_model("./models/model_" + args.d + "_" + args.m + ".h5")
        model.summary()

    if args.d == "fmnist":
        (x_train_total, y_train_total), (x_test, y_test) = fashion_mnist.load_data()
        x_train_total = x_train_total.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        num_train = 50000

        print(infog("y_train len:{}".format(len(y_train_total))))
        print(infog("y_test len:{}".format(len(y_test))))
        # Load pre-trained model.
        model = load_model("./models/model_" + args.d + "_" + args.m + ".h5")
        model.summary()

    if args.d == "cifar100":
        (x_train_total, y_train_total), (x_test, y_test) = cifar100.load_data()
        num_train = 40000
        y_train_total = y_train_total.reshape([y_train_total.shape[0]])
        y_test = y_test.reshape([y_test.shape[0]])

        print(infog("y_train len:{}".format(len(y_train_total))))
        print(infog("y_test len:{}".format(len(y_test))))

        model = load_model("./models/model_" + args.d + "_" + args.m + ".h5")
        model.summary()

    if args.d == "cifar100_coarse":
        (x_train_total, y_train_total), (x_test, y_test) = cifar100.load_data(label_mode='coarse')
        num_train = 40000
        y_train_total = y_train_total.reshape([y_train_total.shape[0]])
        y_test = y_test.reshape([y_test.shape[0]])

        print(infog("y_train len:{}".format(len(y_train_total))))
        print(infog("y_test len:{}".format(len(y_test))))

        model = load_model("./models/model_" + args.d + "_" + args.m + ".h5")
        model.summary()

    elif args.d == "cifar10":
        (x_train_total, y_train_total), (x_test, y_test) = cifar10.load_data()
        num_train = 40000
        y_train_total = y_train_total.reshape([y_train_total.shape[0]])
        y_test = y_test.reshape([y_test.shape[0]])

        print(infog("y_train len:{}".format(len(y_train_total))))
        print(infog("y_test len:{}".format(len(y_test))))

        model = load_model("./models/model_" + args.d + "_" + args.m + ".h5")
        model.summary()

    # data pre-processing
    CLIP_MIN = -0.5
    CLIP_MAX = 0.5
    x_train_total = x_train_total.astype("float32")
    x_train_total = (x_train_total / 255.0) - (1.0 - CLIP_MAX)
    x_test = x_test.astype("float32")
    x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)

    # split original training dataset into training and validation dataset
    x_train = x_train_total[:num_train]
    x_valid = x_train_total[num_train:]
    y_train = y_train_total[:num_train]
    y_valid = y_train_total[num_train:]

    # obtain kde functions and kde inferred classes per class
    fetch_kdes(model, x_train, x_valid, x_test, y_train, y_valid, y_test, layer_names, args)
