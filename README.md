
# Learning to Adapt and Calibrate: Score Distribution Alignment for Few-Shot Uncertainty Prediction in Medical VLMs

### Install

* Install in your environment a compatible torch version with your GPU. For example:

```
conda create -n align python=3.11 -y
conda activate align
pip install torch torchvision torchaudio
```

```
pip install -r requirements.txt
```

### Preparing the datasets
- Configure data paths (see [`./local_data/constants.py`](./local_data/constants.py)).
- Download, and configure datasets (see [`./local_data/datasets/README.md`](./local_data/datasets/README.md)).

## Usage
We present the basic usage here.

(a) Features extraction:
- `python extract_features.py --task Gleason,MESSIDOR`

(b) Conformal prediction (AlignCP):
- `python domain_adapt.py --task Gleason,MESSIDOR --k 16 --alpha 0.10 --ncscore weighted_lac`

You will find the results upon training at [`./local_data/results/`](./local_data/results/).






















