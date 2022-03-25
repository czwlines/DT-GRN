# DT-GRN

Code for "DT-GRN: Heterogeneous Network-based Graph Recurrent Neural Network Model for Drug Combination Prediction".

## Rrequisites
- pyTorch 1.5.1
- Python 3.6.10

## Usage

Dowload the processed datasets from [this site](https://drugcomb.org/) to `data`

### Train

Take "786-0" for example

```
python train_10f.py --data_set "786-0"
```

### Predict

Details in the `predict.ipynb`
