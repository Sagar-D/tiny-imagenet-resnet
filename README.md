# Tiny ImageNet Classification using ResNet18 (TensorFlow)[#](#tiny-imagenet-classification-using-resnet18-tensorflow "Copy link")

This project implements **ResNet18 from scratch using TensorFlow/Keras**  
and trains it on the **Tiny ImageNet dataset** for multi-class image  
classification.

The project focuses on **understanding CNN architectures, TensorFlow  
training pipelines, and model training workflows used in real-world ML  
systems**.

---

# Project Motivation[#](#project-motivation "Copy link")

Instead of using a prebuilt architecture from a library, this project:

-   Implements **ResNet18 architecture manually**
-   Builds a **custom TensorFlow data pipeline**
-   Explores **regularization techniques**
-   Experiments with **training workflows and model checkpointing**

The goal is to gain a deeper understanding of **deep learning system  
design rather than simply training a model**.

---

# Dataset[#](#dataset "Copy link")

Dataset used: **Tiny ImageNet**  
Dataset source: [https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet](https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet)

Dataset statistics:

| Property | Value |
| --- | --- |
| Classes | 200 |
| Training images | 100,000 |
| Validation images | 10,000 |
| Image size | 64 × 64 |

Dataset is downloaded automatically using **kagglehub**.

---

# Model Architecture[#](#model-architecture "Copy link")

The model implements **ResNet18 with residual connections**.

Architecture overview:

Input (64x64x3)

Data Augmentation

-   RandomFlip
-   RandomRotation
-   RandomZoom
-   RandomContrast

Rescaling (1/255)

Initial Convolution Block

-   Conv2D (3x3, 64)
-   BatchNorm
-   ReLU
-   MaxPooling

Residual Stage 1

-   Identity Block (64)
-   Identity Block (64)

Residual Stage 2

-   Projection Block (128)
-   Identity Block (128)

Residual Stage 3

-   Projection Block (256)
-   Identity Block (256)

Residual Stage 4

-   Projection Block (512)
-   Identity Block (512)

Classifier

-   GlobalAveragePooling
-   Dense (200 classes)

Key characteristics:

-   Residual skip connections
-   Batch normalization
-   L2 regularization
-   Global average pooling

---

# TensorFlow Data Pipeline[#](#tensorflow-data-pipeline "Copy link")

A **tf.data pipeline** is implemented to efficiently load and process  
images.

Pipeline flow:

Image Paths + Labels  
    │  
Dataset.from\_tensor\_slices  
    │  
Shuffle  
    │  
Map (image decoding + resizing)  
    │  
Cache  
    │  
Batch  
    │  
Prefetch

Advantages:

-   parallel image decoding
-   efficient GPU training pipeline
-   reduced CPU bottlenecks

---

# Data Augmentation[#](#data-augmentation "Copy link")

Data augmentation is applied **inside the model**, ensuring:

-   augmentation applied only during training
-   validation/inference remain unchanged

Augmentations used:

-   Random horizontal flip
-   Random rotation
-   Random zoom
-   Random contrast

---

# Regularization Techniques[#](#regularization-techniques "Copy link")

To improve generalization:

-   **L2 weight decay**
-   **Data augmentation**
-   **Batch normalization**
-   **Early stopping**

---

# Training Strategy[#](#training-strategy "Copy link")

Optimizer: Adam

Loss: SparseCategoricalCrossentropy

Callbacks used:

-   EarlyStopping
-   ModelCheckpoint

Example:

python

Copy

```python
tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)
```

---

# Model Checkpointing[#](#model-checkpointing "Copy link")

The best model is saved automatically during training:

checkpoints/resnet18\_tiny\_imagenet\_best\_checkpoint.keras

Copy

Features:

-   resume training
-   experiment tracking
-   model versioning

---

# Project Structure[#](#project-structure "Copy link")

tiny-imagenet-resnet

dataset/  
tiny\_imagenet\_data\_loader.py

models/  
[resnet.py](http://resnet.py)

config/  
training\_config.py

checkpoints/  
saved models

[train.py](http://train.py)

[README.md](http://README.md)

---

# Running the Project[#](#running-the-project "Copy link")

## Install dependencies[#](#install-dependencies "Copy link")

text

Copy

```
pip install -r requirements.txt
```

---

## Train the model[#](#train-the-model "Copy link")

shell

Copy

```shell
python -m training.train
```

---

## Resume training[#](#resume-training "Copy link")

shell

Copy

```shell
python -m training.train --resume
```

---

# Current Results[#](#current-results "Copy link")

Current baseline performance:

| Metric | Value |
| --- | --- |
| Training Accuracy | ~55% |
| Validation Accuracy | ~42% |
| Validation Loss | ~2.88 |

---

# Future Improvements[#](#future-improvements "Copy link")

Possible improvements:

-   Learning rate scheduling
-   MixUp / CutMix augmentation
-   Label smoothing
-   Transfer learning using ImageNet pretrained weights
-   ResNet34 / ResNet50 comparison
-   Hyperparameter tuning

---

# Key Learnings[#](#key-learnings "Copy link")

This project helped explore:

-   ResNet architecture internals
-   TensorFlow training pipelines
-   Overfitting vs generalization
-   Model checkpointing workflows
-   GPU training environments
-   Debugging deep learning pipelines

---

# License[#](#license "Copy link")

MIT License