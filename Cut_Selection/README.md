# ✂️ Cut_Selection Module

The `Cut_Selection` directory is responsible for determining the optimal **cut points** in the HuSCF-GAN architecture for distributed training. This is achieved by estimating per-layer computational and communication costs, and using a **Genetic Algorithm** to select the best split points across the network.

---

## 📁 Files Overview

### 🔧 `modelCutter.py`

This module analyzes the model architecture to compute:

- **FLOPs per layer**: Estimated floating point operations required for each layer.
- **Output size per layer (in bytes)**: Represents the communication cost if the layer output is sent across devices.

#### Key Notes:
- The implementation is **architecture-specific** — you'll need to modify both `modelCutter.py` and `CutLayerSelection.py` if you change the model.
- The script **groups layers in blocks of 3**:  
  - Main layer (e.g., Conv2D or Linear)  
  - Normalization layer (e.g., BatchNorm)  
  - Activation function (e.g., ReLU)

  > This is because normalization and activation layers contribute minimally to FLOPs and can be approximated with the main layer for cut point purposes.

- Cuts are selected **between these blocks**, not arbitrary layers, to simplify computation and preserve structure.

---

### 🧬 `CutLayerSelection.py`

This file implements the **Genetic Algorithm** used to select the optimal model cut point based on:

- **Computational latency** (using FLOPs and device compute specs)
- **Transmission latency** (using output size in bytes and network bandwidth)

#### 🧠 Fitness Function:
The genetic algorithm evaluates candidate cut points by calculating **iteration latency**, combining:

1. **FLOPs per block**
2. **Output size (bytes)**
3. **Device profiles** (see below)

It searches for the cut point that minimizes total iteration latency across **all clients**.

---

### 🧩 `profiles.yaml`

This file defines the **device profiles** used by the Genetic Algorithm. Each profile includes:

- `cpu_frequency`: CPU speed in GHz
- `flops_per_cycle`: Device computational capability
- `data_rate`: Communication bandwidth in bytes/sec

You can simulate various environments by:

- Adding new devices
- Modifying existing device specs
- Removing devices for minimal configurations

📍 The Genetic Algorithm will evaluate latency for each client based on these profiles.

---

## ⚠️ Important Notes

- This module is **not dynamic** and is tightly coupled with the model architecture defined in the training pipeline.
- To use a new model architecture, you must:
  1. Modify `modelCutter.py` to reflect the new layer structure.
  2. Adjust the grouping and cut logic if necessary.
  3. Verify consistency in `CutLayerSelection.py`.

Refer to the code comments in both files for guidance on how to adapt the implementation to new models.

---

## 🗂️ Output

The module outputs:
- **Selected cut points for all clients** that minimizes latency
- **Latency values** for each client-device pair

These values are used during the training phase to determine how to split the generator and discriminator between clients and the server.

---

