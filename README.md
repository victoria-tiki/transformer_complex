# Sequence modeling of higher-order wave modes of binary black hole mergers

## Introduction

We introduce a transformer model that predicts the time-series evolution of the pre-merger, merger and ringdown evolution of higher-order wave modes of quasi-circular, spinning, non-precessing binary black hole mergers. Our transformer model takes as input the time-series evolution of the inspiral waveform evolution, as given by the plus and cross polarizations. 

This repository is based on the implementation described in [1], with extensive modifications including a migration from TensorFlow to PyTorch and significant architectural redesign. This repository accompanies the paper [2], where we introduce and analyze the presented model. See the paper for methodological details and discussion of results.

## Installation and Usage
Clone the repository and install the required dependencies to get started:
```
git clone https://github.com/victoria-tiki/transformer_complex.git
cd transformer_complex
```
To train the model using the provided slurm script, run:
```
sbatch submitgpu.slurm
```
Note, the provided models_weights.py file is only required if you need the model to return the attention weights (e.g. for visualization purposes in inference/plot_weights.py).  

For inference, use the following slurm script:
```
sbatch submit_inference.slurm
```
This will run both the inference.py script and later aggregate results over multiple gpus in aggregate_results.py. The resulting hdf5 file can be examined using the provided compute_overlap.ipynb code. A checkpoint file, model.ckpt, is also provided. 

Ensure that you adjust the slurm scripts according to your specific computational environment and requirements.

## Context and Visualizations

For an overview of the paper, results, and context, click [here](https://victoria-tiki.github.io/forecasting_transf/index.html?highlight=visualizations). This page also includes links to interactive visualizations, such as attention maps, extended waveform galleries comparing predicted and true waveforms, and obfuscation studies illustrating how the model uses different parts of the input. For a discussion of attention interpretability and its limitations, see Section 3.6 of our paper.

## References

[1] Khan et alii, *Interpretable AI forecasting for numerical relativity waveforms of quasi-circular, spinning, non-precessing binary black hole mergers*, 2022. [[2110.06968](https://arxiv.org/pdf/2110.06968)]

[2] Tiki et alii, *Sequence modeling of higher-order wave modes of binary black hole mergers*, 2025. [[2409.03833](https://arxiv.org/abs/2409.03833)]


