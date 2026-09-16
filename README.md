# 🧠 Deep Learning Lab & Model Zoo

Welcome to my central repository for deep learning research, model implementations, and experimental benchmarks. This repository serves as a practical workspace where I implement, fine-tune, and evaluate state-of-the-art computer vision and deep learning architectures from scratch and using pre-trained weights.

---

## 📌 Project Overview

This lab contains end-to-end pipelines for image classification, feature extraction, and transfer learning across foundational Convolutional Neural Network (CNN) architectures. 

The primary goal of this repository is to systematically analyze architectural trade-offs—such as parameter efficiency, convergence speed, gradient flow, and inference latency—across different deep learning paradigms.

---

## 🏗️ Supported Architectures & Benchmarks

| Architecture | Paradigm / Innovation | Core Feature | Typical Use Case |
| :--- | :--- | :--- | :--- |
| **VGG16** | Classical Sequential | Small $3 \times 3$ convolutional filters, deep stack | Baseline feature extraction |
| **ResNet-50** | Residual Learning | Skip connections ($F(x) + x$) to solve vanishing gradients | Deep feature representation |
| **DenseNet-121** | Feature Concatenation | Direct layer-to-layer feature reuse, high parameter efficiency | Medical imaging & resource-constrained tasks |
| **InceptionV3** | Multi-Scale Feature Extraction | Parallel $1 \times 1$, $3 \times 3$, $5 \times 5$ branches + auxiliary classifiers | Wide-context image recognition |
| **ResNeXt-50** | Cardinality / Grouped Convolutions | Split-transform-merge with 32 parallel branches | Cost-effective stronger residual baselines |
| **MobileNetV2** | Depthwise Separable Convolutions | Inverted residuals + linear bottlenecks | On-device / edge deployment |
| **EfficientNet-B0** | Neural Architecture Search + Compound Scaling | Uniformly scales width, depth, and resolution | State-of-the-art accuracy / FLOPs trade-off |
| **Xception** | Depthwise Separable Convolutions | Extreme Inception with residual depthwise-separable stacks | Efficient large-scale feature extraction |
| **ViT (Base/16)** | Vision Transformer | Pure self-attention patches instead of convolutions | Modern transformer-based classification |

---

## 🛠️ Tech Stack & Tools

* **Frameworks:** PyTorch / TensorFlow (Keras)
* **Data Processing & Numerical Computing:** NumPy, Pandas, OpenCV
* **Visualization & Metrics:** Matplotlib, Seaborn, Scikit-learn (Confusion Matrix, ROC-AUC)
* **Environment:** Python 3.10+, CUDA, Jupyter Notebooks

---

## 📁 Repository Structure

```text
├── data/                  # Dataset directory (raw & processed)
├── notebooks/             # Exploratory Data Analysis & Experimentation
│   ├── 01_vgg16_baseline.ipynb
│   ├── 02_resnet50_residual_learning.ipynb
│   └── 03_densenet121_benchmarks.ipynb
├── models/                # Custom model architectures and modules
│   ├── vgg.py
│   ├── resnet.py
│   └── densenet.py
├── utils/                 # Data loaders, augmentation, & evaluation metrics
│   ├── dataset.py
│   └── metrics.py
├── train.py               # Main model training and evaluation script
├── requirements.txt       # Project dependencies
└── README.md              # Repository documentation

```


---

## 📈 Key Learnings & Insights

* **Gradient Dynamics:** Observed how residual skip connections in ResNet-50 prevent gradient degradation compared to deeper standard sequential networks like VGG16.
* **Parameter Efficiency:** Evaluated DenseNet-121's ability to achieve competitive accuracy with significantly fewer parameters via feature map concatenation.
* **Optimization:** Fine-tuned hyperparameters using Adam and SGD with learning rate schedulers to stabilize training convergence.

---

## 👤 Author

**Md Arafat Hossain Sani**  
*MERN Stack Developer & Deep Learning Enthusiast*  

* **GitHub:** [@hossain-sani](https://github.com/hossain-sani)  
* **LinkedIn:** [LinkedIn Profile](https://linkedin.com/in/hossain-sani)