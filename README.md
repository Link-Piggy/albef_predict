# albef_predict

This is the project of using pretrained ALBEF model to extract multimodal features. The model and all settings for the prediction are based on original [ALBEF](https://github.com/salesforce/ALBEF) repository. If you are interested in the details of ALBEF, please check their original repo.

## Getting Started
### Installation Requirements
To start with this project, make sure that your environment meets the requirements below:

* python >= 3.8

Run the following command to install required packages.

```bash
pip install -r requirements.txt
```

### Download:
Pre-trained checkpoint [[14M](https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/ALBEF.pth)] / [[4M](https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/ALBEF_4M.pth)]

## Tutorial
### Provide inputs
* image directory storing all images
* csv file that must contain the columns of `product_id`, `img_name`, and `title`

### Set prediction config
All the configuration for predicting is stored in `configs/predict.yaml`. You can modify it according to your needs and refer to the instructions.

### Implement prediction
Run the following command to implement prediction.

```bash
python predict.py --config configs/predict.yaml
```
