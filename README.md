# Implementing Vision Transformers in PyTorch from Scratch on Any Dataset!

![Intro Image (1)](https://github.com/user-attachments/assets/437c8d37-9ae6-49a6-aec2-887c364349c5)

<a target="_blank" href="https://colab.research.google.com/github/ssanya942/MICCAI-Educational-Challenge-2024/blob/master/Implementing_Vision_Transformers_in_PyTorch_from_Scratch.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Introduction-Riding the Classics
Vision Transformers (ViTs) were introduced in 2020 to present a classification approach for images. Having being inspired by the Transformer architecture used prevalently in text classification, ViTs present a robust approach powered by multi-head self-attention to compete with CNNs for image classification/segmentation tasks. There have been numerous implementations of ViTs so far that capture the performance of the model perfectly. However, they are often implementable through external APIs such as [HuggingFace](https://huggingface.co/docs/transformers/en/model_doc/vit) or [Keras](https://keras.io/examples/vision/image_classification_with_vision_transformer/), providing easier computations at the expense of lower architectural transparency. In fact, while these APIs are a Godsend for developers and engineers to facilitate model implementation and fine-tuning at the hit of a button, they are often not the best resort for students starting with Deep Learning. For learners, the best approach to get hands-on deep learning practice is through implementing SOTA models from scratch. In this repository, we have created the most lucid and understandable tutorial for students to implement ViTs from scratch on any dataset of their choice. 

## Installation
Please clone the repository to install all the required files and navigate to the working directory. 
```python
!git clone https://github.com/ssanya942/MICCAI-Educational-Challenge-2024.git
%cd /content/MICCAI-Educational-Challenge-2024
```

To install all the requirements of this implementation, install all the dependencies in the requirements.txt file. 
```python
!pip install -r requirements.txt
```
You can create a new PyTorch environment for this implementation also.

## External Files
For several aspects of this code, including model training, performance curve plotting, and obtaining predictions, the files [engine.py](https://github.com/ssanya942/MICCAI-Educational-Challenge-2024/blob/master/engine.py),[helper_functions.py](https://github.com/ssanya942/MICCAI-Educational-Challenge-2024/blob/master/helper_functions.py), and [predictions.py](https://github.com/ssanya942/MICCAI-Educational-Challenge-2024/blob/master/predictions.py) have been used. Alternatively, they can be accessed from the [PyTorch Deep Learning repository](https://github.com/mrdbourke/pytorch-deep-learning/tree/main/going_modular/going_modular). 

### Model Training
For training the model, the *engine.py* file is used to create the model trainer. First, confirm that you are in the **MICCAI-Educational-Challenge-2024** directory. 
```python
import engine
results = engine.train(model=vit,
                       train_dataloader=train_dataloader,
                       test_dataloader=test_dataloader,
                       optimizer=optimizer,##define optimizer
                       loss_fn=loss_fn, ##define loss function
                       epochs=30,
                       device=device) ## 'cuda', if available

```

### Performance Curve Plotting
To plot accuracy and loss curves after the model is trained, the *helper_functions.py* file is used. 
```python
from helper_functions import plot_loss_curves

# Plot our ViT model's loss curves
plot_loss_curves(results)

```

### Image Prediction
To obtain predictions and probability scores, we use the *predictions.py* file provided in the main repo. 
```python
import predictions
from predictions import pred_and_plot_image
 #Setup custom image path
custom_image_path = "image_pth.png"

# Predict on custom image
pred_and_plot_image(model=vit,
                    image_path=custom_image_path,
                    class_names=class_names)
```

## Conntributors
[Sanya Sinha](https://github.com/ssanya942) and [Nilay Gupta](https://github.com/ngcd04-fa07)

