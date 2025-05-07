# GHONN Models Pytorch
[![Project Status: Active - The project has reached a stable, usable
state and is being actively
developed.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#WIP) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Introduction
**GHONN Models Pytorch** brings advanced neural architectures to your PyTorch projects: Higher Order Neural Units (HONU), Higher Order Neural Networks (HONN), Gated Higher Order Neural Units (gHONU), and Gated Higher Order Neural Networks (gHONN).

✨ **Polynomial neurons at the core:** These models excel at capturing complex, nonlinear relationships—especially when working with polynomial signals. Their adaptable design makes them a strong choice for a wide range of machine learning tasks.

🔗 **Gated variants for extra power:** The gated architectures use a dual HONU neuron setup—one as a dynamic gate, the other as the main predictor—enabling richer and more expressive modeling.

🛠️ **Modular and flexible:** Build your own architectures with ease. Layers can be stacked directly or connected via linear mappings, giving you full control over your network’s structure.

👉 **Curious how it works in practice?** Check out the example notebooks and usage guides included in this repository.

## Requirements & Installation

- **Python:** 3.12 or newer
- **PyTorch:** 2.0.6 or newer

Once the package is released on PyPI, install it with:

```bash
pip install ghonn-models-pytorch
```

Or, to install from source:

```bash
git clone https://github.com/carnosi/ghonn_models_pytorch.git
cd ghonn_models_pytorch
pip install -r requirements.txt
```

## Features

**Neuron Types** ⚡
- **HONU:** The fundamental building block for higher-order modeling. For example, a 2nd order HONU is defined as:

  ![HONU equation](https://latex.codecogs.com/png.image?\dpi{120} \tilde{y}(k)=\sum_{i=0}^{n}\sum_{j=i}^{n}w_{i,j}x_ix_j=\mathbf{w}\cdot\mathrm{col}^{r=2}(\mathbf{x}))

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

**Key Advantages** 🚀
- **Efficient high-order computation:** Redesigned polynomial order assignment enables fast calculations even at high polynomial orders (up to 10th order and beyond) even on CPUs.
- **Seamless PyTorch integration:** All components are implemented as standard PyTorch modules for easy use in your projects.
- **Customizable & modular:** Layers and neurons can be stacked, combined, or adapted to suit a wide range of architectures and tasks.
- **Supports regression & classification:** Suitable for both regression and classification problems.
- **Ready-to-use examples:** Example notebooks and usage guides are included in this repo to help you get started quickly.

## Examples & Usage

- Explore step-by-step Jupyter notebooks in the [examples](./examples/) folder for practical demonstrations and implementation tips.
- Example usage patterns and code snippets will be added soon.

```python
# Example usage coming soon!
```

## Tips & Tricks
* In the case of GHONU based units it is often benefitial to have different initial learning rate between the two neurons.

## Application Examples
The following papers mostly rely on the legacy code instead of the current pytorch version, however the usage target is the same with the pytorch implementation beeing user friendly and computationally efficient compared to the legacy code.

[1] ...


## References
This codebase builds on top of the research conducted in the following papers:
HONU:
[1] ...
GHONU:
[1] ...

## How To Cite
If ghonn_models_pytorch has been useful in your research or work, please consider citing our article:

```plaintext
TBD
```

BibText:
```plaintext
TBD
```

