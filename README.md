# AI-forecasting of higher order wave modes of quasi-circular, spinning, non-precessing binary black hole mergers

## Introduction
We introduce a transformer model that predicts the time-series evolution of the pre-merger, merger and ringdown evolution of higher-order wave modes of quasi-circular, spinning, non-precessing binary black hole mergers. Our transformer model takes as input the time-series evolution of the inspiral waveform evolution, as given by the plus and cross polarizations. Our model is implemented in PyTorch Lightning.

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


