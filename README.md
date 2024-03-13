# Transformer for Complex-Valued Predictions of Binary Black Hole Mergers

## Introduction
This project develops a transformer model designed for forecasting the waveforms of higher-order modes of quasi-circular, spinning, non-precessing binary black hole mergers. Based on the model presented in [arXiv:2110.06968](https://arxiv.org/pdf/2110.06968.pdf), this implementation extends its capabilities to handle complex-valued predictions, specifically targeting the intricate dynamics of binary black hole systems. Our model is re-implemented in PyTorch Lightning, enhancing readability and scalability.

## Installation
Clone the repository and install the required dependencies to get started:
```
git clone https://github.com/yourusername/yourprojectname.git
cd yourprojectname
pip install -r requirements.txt
```

## Usage
To train the model using the provided slurm script, run:
```
sbatch submitgpu.slurm
```

For inference, use the following slurm script:
```
sbatch submit_inference.slurm
```
Ensure that you adjust the slurm scripts according to your specific computational environment and requirements.


