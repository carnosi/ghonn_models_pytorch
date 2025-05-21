<div align="center">

<img src="https://i.ibb.co/4nT0x4HL/GHONN-models-logo.png" alt="logo" width="50%" />

**Python library with polynomial neural networks**

[![Project Status: WIP](https://img.shields.io/badge/repo_status-WIP-<COLOR>?style=for-the-badge&color=yellow)](https://www.repostatus.org/#WIP) [![Read the Docs](https://img.shields.io/readthedocs/gmp?style=for-the-badge&logo=readthedocs&logoColor=white)](https://gmp.readthedocs.io/en/latest/)

[![PyPI](https://img.shields.io/pypi/v/ghonn-models-pytorch?color=red&style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/ghonn-models-pytorch/) [![Python - Version](https://img.shields.io/badge/PYTHON-3.9+-red?style=for-the-badge&logo=python&logoColor=white)](https://pepy.tech/project/ghonn-models-pytorch) [![PyTorch - Version](https://img.shields.io/badge/PYTORCH-2.7+-red?style=for-the-badge&logo=pytorch)](https://pepy.tech/project/ghonn-models-pytorch)

[![License](https://img.shields.io/badge/License-MIT-<COLOR>?style=for-the-badge&color=blue)](https://github.com/carnosi/ghonn_models_pytorch/blob/main/LICENSE)

</div>

**GHONN Models Pytorch** brings advanced neural architectures to your PyTorch projects: Higher Order Neural Units (HONU), Higher Order Neural Networks (HONN), Gated Higher Order Neural Units (GHONU), and Gated Higher Order Neural Networks (GHONN).

✨ **Polynomial neurons at the core:** These models excel at capturing complex, nonlinear relationships—especially when working with polynomial signals. Their adaptable design makes them a strong choice for a wide range of machine learning tasks.

🔗 **Gated variants for extra power:** The gated architectures use a dual HONU neuron setup—one as a dynamic gate, the other as the main predictor—enabling richer and more expressive modeling.

🛠️ **Modular and flexible:** Build your own architectures with ease. Layers can be stacked directly or connected via linear mappings, giving you full control over your network’s structure.

👉 **Curious how it works in practice?** Check out the example notebooks and usage guides included in this repository.

## 📖 [Project Documentation](https://gmp.readthedocs.io/) 📖
Visit [Read The Docs Project Page](https://gmp.readthedocs.io/) or read the following README to know more about Gated Higher Order Neural Network Models Pytorch (GHONN for short) library.

## ✨ Features <a name="features"></a>

- **Polynomial neurons:** Capture complex, nonlinear relationships using higher-order neural units.
- **Gated architectures:** Leverage dual-neuron setups for richer modeling capacity.
- **Modular design:** Easily stack and combine layers for custom architectures.
- **Efficient computation:** Optimized for high-order polynomial calculations, even on CPUs.
- **Seamless PyTorch integration:** All components are standard PyTorch modules.
- **Supports regression & classification:** Flexible for a wide range of ML tasks.
- **Ready-to-use examples:** Example notebooks and guides included.

**Neuron Types** ⚡
- **HONU:** The fundamental building block for higher-order modeling. For example, a 2nd order HONU is defined as:

  ![HONU equation](https://latex.codecogs.com/png.image?\dpi{120}\bg_white\tilde{y}(k)=\sum_{i=0}^{n}\sum_{j=i}^{n}w_{i,j}x_ix_j=\mathbf{w}\cdot\mathrm{col}^{r=2}(\mathbf{x}))

  where:
  - $\tilde{y}(k)$ is the neuron output for input sample $k$
  - $w_{i,j}$ are the learnable weights
  - $x_i, x_j$ are input features
  - $\mathbf{w}$ is the weight vector
  - $\mathrm{col}^{r=2}(\mathbf{x})$ is the column vector of all 2nd order combinations of input features
  - $r$ is the polynomial order

  This structure ensures polynomial relationships between input datapoints and high computation performance.
- **gHONU:** Combines two HONUs—one as a predictor (typically linear activation), the other as a dynamic gate (e.g., `tanh`)—multiplying their outputs for enhanced ability to capture complex patterns.

**Network Layers** 🧩
- **HONN:** Single-layer networks of HONU neurons. Supports both raw outputs for stacking and linear heads for custom output dimensions.
- **gHONN:** Single-layer networks of gHONU neurons, with the same flexible output options as HONN.

**Why Choose GHONN Models?** 🚀
- **Efficient high-order computation:** Optimized for fast polynomial calculations, even at high orders and on CPUs.
- **Flexible & modular:** Easily stack, combine, or adapt layers and neurons for custom architectures.
- **PyTorch-native:** All components are standard PyTorch modules for seamless integration.
- **Versatile:** Supports both regression and classification tasks.
- **Quick start:** Example notebooks and guides included to help you get going fast.

## 🧪 Examples & Usage <a name="examples"></a>

You can find helpful, step-by-step Jupyter notebooks in the [examples](./examples/) folder, which offer practical demonstrations and implementation suggestions.

You may also find the code snippets below useful as a starting point.

**HONU initialization**
```python
import ghonn_models_pytorch as gmp

kwargs = {
    "weight_divisor": 100,  # Divides weights to help with numerical stability
    "bias": True            # Whether to use a bias term in the model
}

# Create a Higher Order Neural Unit (HONU) with 3 inputs and degree 2
honu_neuron = gmp.HONU(
    in_features=3,          # Number of input features
    order=2,                # Degree of the polynomial
    activation="identity",  # Activation function
    **kwargs
)
```

**HONN initialization**
```python
import ghonn_models_pytorch as gmp

kwargs = {
    "weight_divisor": 100,
    "bias": True
}

# Create single HONU based layer - HONN with 4 neurons of different orders and activation functions.
honn_layer = gmp.HONN(
    input_shape=3,                          # Number of input features
    output_shape=2,                         # Number of output features
    layer_size=4,                           # Number of neurons in the layer
    orders=(2, 3)                           # Degree of the polynomials in the layer. If shorter than layer size it works as rolling buffer
    activations=("identity", "sigmoid"),    # Activation functions for the neurons in the layer. If shorter work like a rolling buffer
    output_type="linear",                   # Output type of the layer. Can be "linear" or "sum" or "raw"
    **kwargs
)
```
**Neuron, Layer or Model training as usual**
```python
for i in range(0, data.size(0), batch_size):
    # Get the batch
    batch = data[i:i+batch_size]
    # Forward pass
    output = honn_layer(batch)
    # Compute loss
    loss = criterion(output, target)
    # Backward pass
    loss.backward()
    # Update weights
    optimizer.step()
```

## 💡 Tips & Tricks <a name="tips_n_tricks"></a>
* In the case of GHONU based units it is often benefitial to have different initial learning rate between the two neurons.
* more TBD

## 🛠️ Installation <a name="installation"></a>

**PyPI version:**
```bash
pip install ghonn-models-pytorch
```

**The latest version from GitHub:**
```bash
pip install git+https://github.com/carnosi/ghonn_models_pytorch
```

## 📚 References <a name="references"></a>
This repository is inspired by the foundational research presented in the following papers. While the original studies utilized legacy implementations, this PyTorch-based version offers a more user-friendly and computationally efficient alternative, maintaining the same core objectives and functionality.

**HONU**:
```plaintext
[1] P. M. Benes and I. Bukovsky, “Railway Wheelset Active Control and Stability via Higher Order Neural Units,” IEEE/ASME Transactions on Mechatronics, vol. 28, no. 5, pp. 2964–2975, Oct. 2023, doi: 10.1109/TMECH.2023.3258909.

[2] I. Bukovsky, G. Dohnal, P. M. Benes, K. Ichiji, and N. Homma, “Letter on Convergence of In-Parameter-Linear Nonlinear Neural Architectures With Gradient Learnings,” IEEE Transactions on Neural Networks and Learning Systems, vol. 34, no. 8, pp. 5189–5192, Aug. 2023, doi: 10.1109/TNNLS.2021.3123533.

[3] I. Bukovsky, “Deterministic behavior of temperature field in turboprop engine via shallow neural networks,” Neural Comput & Applic, vol. 33, no. 19, pp. 13145–13161, Oct. 2021, doi: 10.1007/s00521-021-06013-7.

[4] P. M. Benes, I. Bukovsky, M. Vesely, J. Voracek, K. Ichiji, and N. Homma, “Framework for Discrete-Time Model Reference Adaptive Control of Weakly Nonlinear Systems with HONUs,” in Computational Intelligence, C. Sabourin, J. J. Merelo, K. Madani, and K. Warwick, Eds., Cham: Springer International Publishing, 2019, pp. 239–262. doi: 10.1007/978-3-030-16469-0_13.
```
**GHONU**:
```plaintext
[1] O. Budik, I. Bukovsky, and N. Homma, “Potentials of Gated Higher Order Neural Units for Signal Decomposition and Process Monitoring,” Procedia Computer Science, vol. 253, pp. 2278–2287, Jan. 2025, doi: 10.1016/j.procs.2025.01.288.
```

### Our other project

**[AISLEX](https://github.com/carnosi/AISLEX)**: A Python package for Approximate Individual Sample Learning Entropy (LE) anomaly detection. Easily integrate LE-based novelty detection into your neural network workflows, with both Python and JAX implementations.

## 📝 How To Cite <a name="how_to_cite"></a>
If `ghonn_models_pytorch` has been useful in your research or work, please consider citing our article:

```plaintext
Work in progress. Use GHONU (10.1016/j.procs.2025.01.288) for now please.
```

BibText:
```bibtex
Work in progress. Use GHONU (10.1016/j.procs.2025.01.288) for now please.
```
## 📄 License <a name="lisence"></a>

This project is licensed under the terms of the [MIT License](https://github.com/carnosi/ghonn_models_pytorch/blob/main/LICENSE).