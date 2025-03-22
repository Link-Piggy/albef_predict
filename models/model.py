'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

from functools import partial
from models.vit import VisionTransformer
from models.xbert import BertConfig, BertForMaskedLM

import torch
from torch import nn


class ALBEF(nn.Module):
    def __init__(self,config):
        super().__init__()
     
        self.visual_encoder = VisionTransformer(
            img_size=config['resolution'], 
            patch_size=16, 
            embed_dim=768, 
            depth=12, 
            num_heads=12, 
            mlp_ratio=4, 
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
        
        bert_config = BertConfig.from_json_file(config['bert_config'])
        self.text_encoder = BertForMaskedLM.from_pretrained(config['text_encoder'], config=bert_config)            


    def forward(self, image, text):
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)

        text_output = self.text_encoder.bert(
            text.input_ids, 
            attention_mask=text.attention_mask, 
            return_dict=True, 
            mode='text',
        )
        text_embeds = text_output.last_hidden_state        

        output = self.text_encoder.bert(
            encoder_embeds=text_embeds, 
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,      
            return_dict=True,
            mode='fusion',
        )
        return output.last_hidden_state