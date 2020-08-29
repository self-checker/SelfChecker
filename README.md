# SelfChecker
ICSE2021 Submission

Code release of a paper ["Guiding Deep Learning System Testing using Surprise Adequacy"](https://arxiv.org/abs/1808.08444)

## Introduction

This archive includes code for computing Surprise Adequacy (SA) and Surprise Coverage (SC), which are basic components of the main experiments in the paper. Currently, the "run.py" script contains a simple example that calculates SA and SC of a test set and an adversarial set generated using FGSM method for the MNIST dataset, only considering the last hidden layer (activation_3). Layer selection can be easily changed by modifying `layer_names` in run.py.


### Files and Directories

- `run.py` - Script processing SA with a benign dataset and adversarial examples (MNIST and CIFAR-10).

### Command-line Options of run.py

- `-d` - The subject dataset (either mnist or cifar). Default is mnist.

## How to Use

Our implementation is based on Python 3.5.2, Tensorflow 1.9.0, Keras 2.2, Numpy 1.14.5. Details are listed in `requirements.txt`.

This is a simple example of installation and computing LSA or DSA of a test set and FGSM in MNIST dataset.

```bash
# install Python dependencies
pip install -r requirements.txt

# train a model
python train_model.py -d mnist

# calculate LSA, coverage, and ROC-AUC score
python run.py -lsa

# calculate DSA, coverage, and ROC-AUC score
python run.py -dsa
```

## Supplementary Experiments Results

|   1   |  2    |  3    |
| ---- | ---- | ---- |
|    0  |    0  |  0    |
|    1  |    1  |   1   |
|    2  |    2  |   2   |



## Notes

- If you encounter `ValueError: Input contains NaN, infinity or a value too large for dtype ('float64').` error, you need to increase the variance threshold. Please refer to the configuration details in the paper (Section IV-C).

- Images were processed by clipping its pixels in between -0.5 and 0.5.
- If you want to select specific layers, you can modify the layers array in `run.py`.
- Coverage may vary depending on the upper bound.
- For speed-up, use GPU-based TensorFlow.
- [All experimental results](https://coinse.github.io/sadl/)
## References

- [DeepXplore](https://github.com/peikexin9/deepxplore)
- [DeepTest](https://github.com/ARiSE-Lab/deepTest)
- [Detecting Adversarial Samples from Artifacts](https://github.com/rfeinman/detecting-adversarial-samples)
- [Characterizing Adversarial Subspaces Using Local Intrinsic Dimensionality](https://github.com/xingjunm/lid_adversarial_subspace_detection)
