# SVM Face Identification

*[中文](README_CN.md)*

A face recognition **baseline** that compares two classifiers on raw pixel
data: a from-scratch backpropagation neural network written in NumPy, and an
RBF-kernel support vector machine from scikit-learn.

> **Status: archived.** This repository is a course project kept for
> reference. The code runs, is tested, and is documented, but the approach
> (classifying raw pixels) is a baseline rather than a competitive face
> recognition method. See [Limitations](#limitations) before reusing it.

## Table of contents

- [What this repository is](#what-this-repository-is)
- [Quick start — no dataset required](#quick-start--no-dataset-required)
- [Dataset acquisition and licensing](#dataset-acquisition-and-licensing)
- [Using your own dataset](#using-your-own-dataset)
- [Architecture](#architecture)
- [Installation](#installation)
- [Command line reference](#command-line-reference)
- [Output artifacts](#output-artifacts)
- [Reported results](#reported-results)
- [Reproducibility](#reproducibility)
- [Testing](#testing)
- [Changes in this version](#changes-in-this-version)
- [Limitations](#limitations)
- [License](#license)

## What this repository is

The original project trained a 500-class face classifier on CASIA-FaceV5 and
compared a neural network against an SVM. Two things limited its usefulness:
the dataset cannot be redistributed, so nobody could run it, and the reported
accuracies came from a single train/test split with hyper-parameters chosen by
hand.

This version keeps the original algorithm and the original results, and fixes
the parts that made the project hard to reuse:

- **The pipeline runs without any download.** `--synthetic` generates a small
  labelled dataset on the fly and runs the whole pipeline in seconds.
- **Every reported number is a mean with a standard deviation** across
  cross-validation folds, not a single point estimate on one split.
- **The train and test splits share one class mapping.** In the original code
  each split built its own mapping independently, so the two could disagree on
  which index meant which person and the reported accuracy became noise.
- **The whole pipeline is covered by tests** that need neither the dataset nor
  a GPU.

## Quick start — no dataset required

```bash
git clone https://github.com/ZajacMo/SVM-Face-Identify.git
cd SVM-Face-Identify

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python main.py --synthetic --epochs 200
```

This generates a synthetic dataset, splits it, trains both models, evaluates
them, and writes every artifact under `results/`. It needs no dataset, no
network access and no GPU.

The synthetic classes are distinguished by colour, so the accuracies it
produces are high. **They say nothing about face recognition** — the run exists
to prove the pipeline works.

## Dataset acquisition and licensing

**CASIA-FaceV5 cannot be redistributed, and this repository does not include
it.** Access requires a signed licence agreement with the institute that
operates the CASIA face database.

To obtain it:

1. Visit the CASIA face database portal at
   <http://www.cbsr.ia.ac.cn/users/jianquan/english/CASIA-FaceV5.html>.
2. Download and complete the licence agreement form.
3. Submit the signed agreement to the institute by the stated email address.
4. Wait for approval. The dataset is normally delivered as an archive over
   FTP once the agreement has been processed.

Place the extracted dataset so that the project root contains a directory with
one sub-directory per person, each holding that person's images:

```
64_CASIA-FaceV5/
    000/
        000_0.bmp
        000_1.bmp
        ...
    001/
        001_0.bmp
        ...
```

Then run the pipeline against it:

```bash
python main.py --dataset 64_CASIA-FaceV5 --tune
```

**Do not commit the dataset.** It is large, it is not yours to redistribute,
and the `.gitignore` already excludes `64_CASIA-FaceV5/`.

## Using your own dataset

Any dataset with the same one-directory-per-class layout works. Nothing in the
code is specific to CASIA: the class name is read from the directory name, and
`.bmp`, `.png` and `.jpg` images are all accepted.

```bash
python main.py --dataset /path/to/my_dataset --test-size-per-class 2
```

Class names may be numeric or non-numeric; they are sorted numerically when
they parse as integers and lexicographically otherwise.

## Architecture

```
main.py                       Command line entry point and pipeline orchestration
src/
    data/
        data_split.py         Train/test splitting and the synthetic generator
        dataset.py            Shared class mapping and manifest loading
    training/
        train.py              Backpropagation network training
        svm_classifier.py     SVM training, tuning and cross-validation
    testing/
        test.py               Evaluation of the neural network on a manifest
        compare_models.py     Head-to-head comparison of both models
        metrics.py            Accuracy, macro precision/recall/F1, confusion matrix
    plotting/
        plot_training_history.py  Loss curve, accuracy, confidence, confusion matrix
    utils/
        utils.py              Image preprocessing and the NumPy neural network
tests/                        pytest suite, no dataset required
```

The two models are deliberately independent so they can be compared fairly:

- **BP neural network** (`src/utils/utils.py`) — one hidden layer with a
  sigmoid activation, a softmax output layer and cross-entropy loss, trained
  with mini-batch gradient descent. Implemented with NumPy only.
- **SVM** (`src/training/svm_classifier.py`) — `sklearn.svm.SVC` with an RBF
  kernel, optionally tuned by cross-validated grid search.

Both consume the same `src/data/dataset.py` loader, which guarantees that the
train and test splits agree on the class indices.

## Installation

Python 3.10 or newer is required.

```bash
pip install -r requirements.txt
```

The runtime dependencies are `numpy`, `Pillow`, `matplotlib`, `seaborn` and
`scikit-learn`. `pytest` is needed only to run the test suite.

## Command line reference

| Flag | Default | Description |
| --- | --- | --- |
| `--dataset` | `64_CASIA-FaceV5` | Root directory holding one sub-directory per class |
| `--synthetic` | off | Generate a synthetic dataset instead of reading `--dataset` |
| `--synthetic-classes` | `12` | Number of classes to generate in synthetic mode |
| `--label-dir` | `labels` | Directory holding the train/test manifests |
| `--output-dir` | `results` | Directory for models, metrics and figures |
| `--image-size` | `32` | Side length of the square images fed to the models |
| `--epochs` | `1000` | Training epochs for the neural network |
| `--learning-rate` | `0.01` | Step size of the gradient update |
| `--batch-size` | `32` | Samples per gradient step |
| `--hidden-size` | `128` | Units in the hidden layer |
| `--test-size-per-class` | `1` | Images reserved for the test set per class |
| `--seed` | `42` | Seed for all random operations |
| `--cv` | `5` | Folds for cross-validation and hyper-parameter search |
| `--tune` | off | Run a cross-validated grid search for the SVM |
| `--steps` | all | Comma separated subset of `split,train,test,svm,compare,plot` |

`--steps` lets you run a single stage, for example to re-evaluate an existing
model:

```bash
python main.py --dataset 64_CASIA-FaceV5 --steps test,plot
```

## Output artifacts

| Path | Contents |
| --- | --- |
| `labels/classes.json` | Ordered list of class names |
| `labels/train_labels.json` | Training manifest of `{path, label}` entries |
| `labels/test_labels.json` | Test manifest of `{path, label}` entries |
| `results/face_model.pkl` | Trained neural network (**not committed**) |
| `results/svm_face_model.pkl` | Trained SVM (**not committed**) |
| `results/class_labels.pkl` | Class mapping of the neural network (**not committed**) |
| `results/training_history.pkl` | Loss history (**not committed**) |
| `results/summary.json` | Run summary including the measured accuracy |
| `results/svm_metrics.json` | SVM metrics, cross-validation folds and best parameters |
| `results/plots/test_results.json` | Per-sample predictions of the neural network |
| `results/plots/test_log.txt` | Human readable evaluation log |
| `results/plots/*.png` | Loss curve, accuracy, confidence distribution, confusion matrix |
| `results/comparison/comparison_results.json` | Accuracy of both models side by side |

Trained models are excluded from version control on purpose. The original SVM
model alone was 61 MB, which made the repository unwieldy and left the working
tree dirty on every re-run. The pipeline regenerates all of them in seconds.

## Reported results

### Original run on CASIA-FaceV5 (500 classes)

These are the numbers from the original run of this project, recorded in the
committed artifacts in `results/`. They were produced by the code **before**
the fixes described in the next section, on a single train/test split, and are
reproduced here for historical reference only. They have not been re-measured,
because the dataset is not redistributable and is not present in this
repository.

| Model | Test accuracy | Source |
| --- | --- | --- |
| BP neural network | **0.702** (351 / 500 correct) | `results/plots/test_results.json` |
| SVM | **0.620** | `results/comparison/comparison_results.json` |

Split details, from the committed manifests:

- 500 classes (`labels/classes.json`), 2000 training images and 500 test
  images.
- The original split reserved **one test image per class**, so each of the 500
  test images came from a different person.

Both figures come from a single split with no cross-validation and no
hyper-parameter search, so they carry an unknown margin of error. The
cross-validated figures the current code produces are the ones to trust.

### Synthetic dataset run

Measured on this machine with the command below, using the synthetic dataset
that the current code generates. **These numbers reflect colour-separated
synthetic shapes, not faces**, and are included only to show what a healthy
run looks like:

```bash
python main.py --synthetic --synthetic-classes 12 --epochs 200 \
    --image-size 32 --cv 5 --tune
```

| Metric | Value |
| --- | --- |
| SVM cross-validated accuracy | 1.0000 +/- 0.0000 |
| SVM test accuracy | 1.0000 |
| SVM best parameters | `C=1.0`, `gamma='scale'`, `kernel='rbf'` |
| BP neural network test accuracy | 0.8333 (10 / 12) |

## Reproducibility

Every random operation is seeded. `--seed` (default `42`) controls the
train/test split, the network weight initialisation, the batch shuffling and
the cross-validation fold assignment. Two runs with the same seed and the same
flags produce the same split and the same loss history.

Cross-validation uses `StratifiedKFold` with `shuffle=True` and a fixed seed,
so the fold assignment is reproducible too.

## Testing

```bash
pip install pytest
python -m pytest tests/ -v
```

The suite covers the dataset splitting, the class mapping, both models and the
end-to-end command line pipeline. **It never touches CASIA-FaceV5**: a small
synthetic dataset is generated as a fixture, so the tests run in seconds on a
laptop and on CI.

Continuous integration runs the suite on Ubuntu and Windows with Python 3.10
and 3.12, and additionally smoke-tests the synthetic pipeline
(`.github/workflows/ci.yml`).

## Changes in this version

Correctness fixes:

- **The train and test splits now share one class mapping.** Previously each
  manifest built its own `label -> index` mapping, so the two splits could
  disagree and the reported accuracy was effectively meaningless.
- **The train/test split is seeded.** It was previously drawn with
  `random.sample` without a seed, so every run produced a different split and
  no result was reproducible.
- **The split no longer filters the training list by membership.** The
  original `train_images = [img for img in images if img not in test_images]`
  was an O(n²) scan that could drop entries with duplicate filenames.
- **Class names no longer have to be numeric.** `sorted(..., key=int)` raised
  `ValueError` on any non-numeric label, which made the code unusable with a
  custom dataset.
- **The sigmoid no longer overflows.** `1 / (1 + np.exp(-x))` overflowed for
  large negative inputs and produced `nan` gradients. It now uses the
  numerically stable two-branch formulation.
- **Paths are no longer hardcoded.** The evaluation code mixed `plots/` and
  `results/plots/`, and used a hardcoded empty project root, which made image
  paths depend on the current working directory.
- **Images are no longer read through `freopen`-style stream redirection.**
  The manifests and the models are read with ordinary file APIs.

Methodological improvements:

- The SVM hyper-parameters are searched with a cross-validated grid search
  (`--tune`) instead of being hardcoded.
- Results are reported as a mean and a standard deviation across folds, with
  macro precision, recall and F1 and a confusion matrix alongside accuracy.

Engineering:

- An `argparse` CLI with a documented `--steps` flag.
- English docstrings and type hints across the public API.
- A pytest suite and a GitHub Actions workflow.
- Model binaries are no longer committed.

## Limitations

Read this before reusing the code.

- **Raw pixels are not face embeddings.** Both models consume the flattened
  RGB image. Real face recognition systems use an embedding model (a ResNet or
  a face network) that maps an image to a compact descriptor, which is both
  far more accurate and dramatically cheaper to classify. The numbers in this
  repository reflect the baseline, not what face recognition can achieve.
- **The original test set is one image per person.** With 500 classes and a
  single held-out image each, the measured accuracy has a large variance and
  is not a stable estimate of generalisation.
- **The original accuracy figures are not cross-validated.** They come from
  one split. Use the cross-validated numbers the current code produces.
- **The dataset is not in the repository and cannot be added to it.** Every
  result on CASIA-FaceV5 must be produced locally by someone who has signed
  the licence agreement.
- **The synthetic dataset is a plumbing test only.** Its accuracy reflects
  colour-separated shapes, not faces.
- **The neural network is a single hidden layer.** It has no regularisation,
  no early stopping and no learning-rate schedule, so it overfits readily on
  a small dataset.
- **The `pickle` model files are not safe to load from untrusted sources.**
  Pickle deserialisation executes arbitrary code.

## License

MIT. See [LICENSE](LICENSE).

The CASIA-FaceV5 dataset is **not** covered by this license. It is
redistributable only to licensees under a separate agreement with the
instituting institute.
