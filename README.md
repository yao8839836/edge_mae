# Edge MAE

The implementation of Edge MAE in our paper: 

Friend Recall in Online Games via Pre-training Edge Transformers.

The repository is modified from [a MAE implementation](https://github.com/IcarusWizard/MAE) and tested on Python 3.7.


## Installing requirement packages

```bash
pip install -r requirements.txt
```

## Data

(1) Due to privacy reason, we could not provide real world data, but we prepare synthetic data in the same format in ./data.

(2) ./data/train and ./data/val are for edge classification training and validation.

(3) ./data/unlabeled are for Edge MAE pre-traininig.

## How to run
 
### 1. Edge MAE pre-training


```shell
python mae_pretrain.py --model_path results/edge-mae.pt
```


### 2. Edge Transformer fine-tuning

```shell
python train_classifier.py --pretrained_model_path results/edge-mae.pt --output_model_path results/transformer2L_mae.pt
```

### 3. Train CNN (ConvKB) and Bilinear for link prediction


#### CNN (ConvKB)

```shell
python link_cnn.py
```

#### Bilinear

```shell
python link_bilinear.py
```
