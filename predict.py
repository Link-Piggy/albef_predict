# -*- coding: utf-8 -*-
'''
This script extracts image and text multimodal features.
'''

import argparse
import loguru
import json
import torch
import ruamel.yaml as yaml

from tqdm import tqdm

from data import get_pred_dataloader
from models.model import ALBEF
from models.tokenization_bert import BertTokenizer

logger = loguru.logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', 
        default='/Users/dachuan/projects/mjj/albef_predict/configs/predict.yaml', 
        help='config file for prediction'
    )
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    logger.info(f"Using {device} device")

    # Get data.
    logger.info("Preparing image & text inference dataset.")
    dataloader = get_pred_dataloader(config)

    # Initialize the model.
    logger.info("Initializing the model.")
    model = ALBEF(config)
    model = model.to(device)

    # Resume from the checkpoint.
    checkpoint = torch.load(config['checkpoint'], map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(f"Load checkpoint from {config['checkpoint']}")

    # Make inference.
    tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])
    write_cnt = 0
    with open(config['feat_output_path'], "w") as fout:
        model.eval()
        with torch.no_grad():
            for (ids, images, texts) in tqdm(dataloader):
                # load data
                images = images.to(device)
                text_input = tokenizer(
                    texts, padding='longest', truncation=True, max_length=64, return_tensors="pt"
                ).to(device)
                # forward
                features = model(images, text_input)
                features = features[:,0,:] if config['only_cls_embed'] else features
                for id, feature in zip(ids.tolist(), features.tolist()):
                    fout.write("{}\n".format(json.dumps({"product_id": id, "feature": feature})))
                    write_cnt += 1
    logger.info(f"{write_cnt} features are stored in {config['feat_output_path']}")