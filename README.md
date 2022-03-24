# DT-GRN

Code for "DT-GRN: Heterogeneous Network-based Graph Recurrent Neural Network Model for Drug Combination Prediction".

## Usage

Dowload the processed datasets from [this site](https://drugcomb.org/) to `data`

### Train

Take "786-0" for example

```
CUDA_VISIBLE_DEVICES=0 python train_10f.py --data_set "786-0"
```

### Predict

Details in the `predict.ipynb`
