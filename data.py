import re
import os
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def pre_caption(caption: str, max_words: int):
    """processes and cleans up a text string"""

    # Remove specific punctuation and normalize the text
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    # Normalize whitespace
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )

    # Remove trailing newline and extra spaces
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    # Truncate the caption to a maximum number of words
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])
    
    return caption


class PredDataset(Dataset):
    def __init__(self, config):
        self.df = pd.read_csv(config['data'], header=0)
        for col in ('product_id', 'img_name', 'title'):
            assert col in self.df.columns, f"Column '{col}' does not exist in {config['data']}"

        self.max_words = config['max_words']
        self.image_dir = config['image_dir']
        self.transform = self._build_transform(config['resolution'])

    def _build_transform(self, resolution):
        normalize = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), 
            (0.26862954, 0.26130258, 0.27577711)
        )
        return transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        data = self.df.iloc[index]

        # get product_id
        id = data.product_id

        # get image tensor
        image = Image.open(
            os.path.join(self.image_dir, data.img_name)
        ).convert('RGB')
        image = self.transform(image)

        # get text of product title
        text = pre_caption(data.title, self.max_words)
        return id, image, text


def get_pred_dataloader(config):
    dataset = PredDataset(config)
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        pin_memory=False,
        shuffle=False,
        drop_last=False,
    )
    return dataloader