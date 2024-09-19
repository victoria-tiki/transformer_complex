# AI-forecasting of higher order wave modes of quasi-circular, spinning, non-precessing binary black hole mergers

## Introduction
We introduce a transformer model that predicts the time-series evolution of the pre-merger, merger and ringdown evolution of higher-order wave modes of quasi-circular, spinning, non-precessing binary black hole mergers. Our transformer model takes as input the time-series evolution of the inspiral waveform evolution, as given by the plus and cross polarizations. Our model is implemented in PyTorch Lightning.

## Installation
Clone the repository and install the required dependencies to get started:
```
git clone https://github.com/victoria-tiki/tranformer_complex.git
cd tranformer_complex
```

## Usage
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


