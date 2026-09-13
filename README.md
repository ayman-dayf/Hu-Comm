# HuSCF-GAN: Distributed Conditional GAN with Heterogeneous U-shape Split Clustered Federated Learning

## 📌 Abstract

Federated Learning has gained increasing attention for its ability to enable multiple nodes to
collaboratively train machine learning models without sharing their raw data. At the same
time, Generative AI—particularly Generative Adversarial Networks (GANs)—have achieved
remarkable success across a wide range of domains, such as healthcare, security, and Image
Generation. However, training generative models typically requires large datasets and signif-
icant computational resources, which are often unavailable in real-world settings. Acquiring
such resources can be costly and inefficient, especially when many underutilized devices—
such as IoT devices and edge devices—with varying capabilities remain idle. Moreover,
obtaining large datasets is challenging due to privacy concerns and copyright restrictions,
as most devices are unwilling to share their data. To address these challenges, we propose
a novel approach for decentralized GAN training that enables the utilization of distributed
data and underutilized, low-capability devices while not sharing data in its raw form. Our
approach is designed to tackle key challenges in decentralized environments, combining
KLD-weighted Clustered Federated Learning to address the issues of data heterogeneity
and multi-domain datasets, with Heterogeneous U-Shaped split learning to tackle the chal-
lenge of device heterogeneity under strict data sharing constraints—ensuring that no labels
or raw data, whether real or synthetic, are ever shared between nodes. Experimental re-
sults shows that our approach demonstrates consistent and significant improvements across
key performance metrics, where it achieves 1.1×—2.2× higher image generation scores,
an average 10% boost in classification metrics (up to 50% in multi-domain non-IID set-
tings), in much lower latency compared to several benchmarks.


---

## 🧠 Overview

**HuSCF-GAN** introduces a **distributed Conditional GAN (cGAN)** training framework leveraging a novel **Heterogeneous U-shape Split Clustered Federated Learning** approach. It is designed to address the following challenges in distributed machine learning:

- **Data heterogeneity**
- **Device heterogeneity**
- **Resource-constrained environments**
- **Multi-domain datasets**
- **Data sharing constraints**

This framework distributes the model across clients and a central server. The split point (cut) in the model is optimally selected using a **Genetic Algorithm**, based on each client’s device profile (e.g., CPU frequency, FLOPs per cycle, data rate).



---



### ▶️ Run Training

To start the training process:

`python train.py`

> **Note:** Run this command from inside the project directory.
> **Note:** Different runs may yield different but close results.



The output of the training—Stroed in `Results/` directory—will look like this:

Below are the generation score plots for each domain for Scenario 6 (Multi-Domain IID):

<table>
  <tr>
    <td align="center">
      <img src="./Results/Scenario_6/plots/mnist_scores.png" width="250"/><br>
      <b>MNIST</b>
    </td>
    <td align="center">
      <img src="./Results/Scenario_6/plots/fmnist_scores.png" width="250"/><br>
      <b>Fashion-MNIST</b>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="./Results/Scenario_6/plots/kmnist_scores.png" width="250"/><br>
      <b>KMNIST</b>
    </td>
    <td align="center">
      <img src="./Results/Scenario_6/plots/notmnist_scores.png" width="250"/><br>
      <b>NotMNIST</b>
    </td>
  </tr>
</table>

Below are how Classification Metrics and Latency appear inside the csv file, This example is the results for Scenario 6:

| Dataset    | Accuracy | Precision | Recall | F1 Score | FPR |
|------------|----------|-----------|--------|----------|-------------|
| MNIST      | 95.94     | 95.97      | 96.96   | 95.93    | 0.45        |
| FashionMNIST | 81.94  | 82.26      | 81.94   | 81.98    | 2.01        |
| KMNIST     | 72.91     | 74.31      | 72.91   | 73.09    | 3.01        |
| NotMNIST   | 88.3     | 88.54      | 88.28   | 88.32    | 1.3        |


| Latency (s) | 
|------------|
| 7.68      | 

---
# HuSCF-GAN: Federated Generative Learning in Multi-Domain Environments

A distributed generative AI framework that leverages split learning and GANs for training across heterogeneous clients with domain-specific data. The architecture adapts using a Genetic Algorithm to determine the optimal model cut based on device capabilities.

---

## 📁 Scenarios

| Scenario Type     | Domains                            | Distribution Type               |
|-------------------|------------------------------------|---------------------------------|
| Single Domain     | MNIST                              | IID / Non-IID                   |
| Double Domain     | MNIST + FMNIST                     | IID / Non-IID / Highly Non-IID  |
| Multi-Domain      | MNIST, FMNIST, KMNIST, NotMNIST    | IID                             |

---

## ⚙️ Configuration

Training scenarios are defined in `configs.yaml`, where you can customize:

- Dataset combinations  
- Data distribution types (IID / Non-IID / Highly Non-IID)  
- Number of clients per domain  
- Training settings and hyperparameters  

### Scenario Setup Guidelines

| Scenario Number | Description                        | Client Dataset Setup                                                                 |
|-----------------|------------------------------------|----------------------------------------------------------------------------------------|
| 1               | Single Domain - IID                | All clients from dataset 0 (e.g., MNIST) except one                                   |
| 2               | Single Domain - Non-IID            | Same as Scenario 1 but with `scenario: 2`                                        |
| 3               | Double Domain - IID                | Clients from two datasets (e.g., MNIST + FMNIST); set others to zero                  |
| 4               | Double Domain - Non-IID            | Same as Scenario 3, with `scenario: 4`                                           |
| 5               | Double Domain - Highly Non-IID     | Same setup as Scenario 3 with  `scenario: 5`                       |
| 6               | Multi-Domain - IID                 | Non-zero number of clients for all datasets (MNIST, FMNIST, KMNIST, NotMNIST)         |

---

## 🔧 Model Splitting & Device Profiles

The generator and discriminator are split across clients and the server. A **Genetic Algorithm** optimizes the cut based on device characteristics.

Check the `Cut_Selection/` directory for more explanation.

### 📱 Device Profiles

Device performance is defined in `Cut_Selection/profiles.yaml`:

- `cpu_frequency`: CPU clock speed (e.g., GHz)  
- `flops_per_cycle`: Computational capacity  
- `data_rate`: Communication bandwidth  

You can add, edit, or remove profiles to simulate different hardware configurations.

---

## 📊 Evaluation Metrics

Metrics are implemented in the `Metrics/` directory. The framework evaluates:

- **Diversity & Coverage**  
  Uses inception-like scores with dataset-specific classifiers to assess realism and variety of generated images.

- **Classification Accuracy**  
  Measures how well generated images preserve class-specific features.

- **Latency**  
  End-to-end system delay, including computation and communication overhead.

Check `Metrics/` directory for more explanation.

---

## 🧠 Supported Datasets

This project supports the following standard image classification datasets:

- **MNIST**  
- **Fashion-MNIST (FMNIST)**  
- **Kuzushiji-MNIST (KMNIST)**  
- **NotMNIST**

These are used across various federated learning scenarios (single, double, multi-domain).

---

## 📂 Outputs

- **Results Directory**: Contains all experiment results, including Classification Metrics, Image Generation Scores Plots and Latency.  
- **Visualization Directory**: Stores generated images across training rounds for visual analysis.

---

### 🛠️ Important Note on Architecture Customization

> ⚠️ **This implementation is tailored to a specific architecture.**  
> It is **not fully dynamic or plug-and-play** for arbitrary model architectures. If you wish to modify or adapt the model architecture (e.g., changing the generator/discriminator structure), you must:
>
> - Carefully follow the **code comments** provided throughout the codefiles.
> - The Model architecture should be changed (if needed) in the `models.py` file as well as the `Cut_Selection/modelCutter.py`, as well as the associated variables in both `Cut_Selection/modelCutter.py` and `Cut_Selection/CutLayerSelection.py` according to the comments. 

## 🌐 Paper Website

The official paper website for **“A Distributed Generative AI Approach for Heterogeneous Multi-Domain Environments under Data Sharing Constraints”** is available at:

🔗 https://distributed-gen-ai.github.io/huscf-gan.github.io/

The website contains:
- An overview of the proposed method and problem setting  
- Figures and qualitative results from the paper  
- Supplementary explanations and resources  
- Links to the arXiv preprint and the Transactions on Machine Learning Research (TMLR) version


## 📚 Citation

If you use HuSCF-GAN in your research or build upon it, please cite:

```bibtex

@article{tawfilis2026a,
title={A Distributed Generative {AI} Approach for Heterogeneous Multi-Domain Environments under Data Sharing constraints},
author={Youssef Tawfilis and Hossam Amer and Minar El-Aasser and Tallal Elshabrawy},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2026},
url={https://openreview.net/forum?id=rpbL7pfPYH},
note={}
}

@misc{tawfilis2025distributedgenerativeaiapproach,
  title        = {A Distributed Generative {AI} Approach for Heterogeneous Multi-Domain Environments under Data Sharing Constraints},
  author       = {Tawfilis, Youssef and Amer, Hossam and El-Aasser, Minar and Elshabrawy, Tallal},
  year         = {2025},
  eprint       = {2507.12979},
  archivePrefix= {arXiv},
  primaryClass = {cs.LG},
  url          = {https://arxiv.org/abs/2507.12979},
  note         = {Accepted and published in Transactions on Machine Learning Research (TMLR), 2026.}
}

