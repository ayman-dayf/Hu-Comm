## 📊 Evaluation Methodology

The evaluation of HuSCF-GAN is performed using **three core metrics** designed to assess the quality, utility, and efficiency of the generated data in federated, multi-domain environments.

---

### 1. 🧠 Image Generation Score

We adopt a variant of the **Inception Score (IS)**, but instead of the standard Inception v3 model, we use **dataset-specific classifiers** that are pretrained on the respective datasets (MNIST, FMNIST, KMNIST, NotMNIST).

- This method provides a more **domain-relevant measure** of the diversity and realism of generated samples.
- The scores are computed by feeding synthetic data into these pretrained classifiers and evaluating the entropy across predictions.

📁 **Location**:  
`Metrics/evaluation.py`  
- Loads pretrained classifiers (present in the same directory).
- Computes the dataset-specific inception-like score for each domain.

---

### 2. ✅ Classification Metrics

To assess the **usability** of generated images for downstream tasks, we calculate:

- **Accuracy**
- **Precision (macro)**
- **Recall (macro)**
- **F1-score (macro)**
- **False Positive Rate (FPR macro)**

#### 📌 Procedure:
After the GAN finishes training:
- A **CNN classifier is trained solely on 30,000 synthetic images per domain**.
- This classifier is then evaluated on the **real test data** for each corresponding domain.
- The result reflects how well the generated data preserves task-relevant features.

📁 **Location**:  
`Metrics/evaluation.py`  
- Contains the CNN training logic on synthetic data.
- Computes metrics on real test data per domain.

---

### 3. ⏱️ Latency Measurement

Latency reflects the **system overhead** during training, including:

- Model computation
- Communication delays
- Device heterogeneity

This is calculated using a **Genetic Algorithm** that optimizes the generator/discriminator cut point across clients and server based on device profiles.

📁 **Location**:  
`Cut_Selection/CutLayerSelection.py`  
- Implements the Genetic Algorithm used for model splitting.
- Evaluates and returns latency for the final selected architecture.

---

