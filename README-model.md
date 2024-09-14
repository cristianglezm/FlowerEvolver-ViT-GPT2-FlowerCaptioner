---
language:
- en
tags:
- image-to-text
- image-captioning
license: apache-2.0
base_model: nlpconnect/vit-gpt2-image-captioning
widget:
- src: >-
    https://huggingface.co/datasets/cristianglezm/FlowerEvolver-Dataset/resolve/main/flowers/001.png
  example_title: Flower 1
- src: >-
    https://huggingface.co/datasets/cristianglezm/FlowerEvolver-Dataset/resolve/main/flowers/002.png
  example_title: Flower 2
- src: >-
    https://huggingface.co/datasets/cristianglezm/FlowerEvolver-Dataset/resolve/main/flowers/003.png
  example_title: Flower 3
datasets:
- cristianglezm/FlowerEvolver-Dataset
metrics:
- rouge
pipeline_tag: image-to-text
library_name: transformers
---

# ViT-GPT2-FlowerCaptioner

This model is a fine-tuned version of [nlpconnect/vit-gpt2-image-captioning](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning) on the [FlowerEvolver-dataset](https://huggingface.co/datasets/cristianglezm/FlowerEvolver-Dataset) dataset.
It achieves the following results on the evaluation set:
- Loss: 0.3075
- Rouge1: 66.3702
- Rouge2: 45.5642
- Rougel: 61.401
- Rougelsum: 64.0587
- Gen Len: 49.97

## sample running code

with python

```python
from transformers import pipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FlowerCaptioner = pipeline("image-to-text", model="cristianglezm/ViT-GPT2-FlowerCaptioner", device=device)
FlowerCaptioner(["flower1.png"])
# A flower with 12 petals in a smooth gradient of green and blue. 
# The center is green with black accents. The stem is long and green.
```

with javascript

```javascript
import { pipeline } from '@xenova/transformers';

// Allocate a pipeline for image-to-text
let pipe = await pipeline('image-to-text', 'cristianglezm/ViT-GPT2-FlowerCaptioner-ONNX');

let out = await pipe('flower image url');
// A flower with 12 petals in a smooth gradient of green and blue. 
// The center is green with black accents. The stem is long and green.
```

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 4
- eval_batch_size: 4
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 500
- num_epochs: 3

### Training results

| Training Loss | Epoch | Step | Validation Loss | Rouge1  | Rouge2  | Rougel  | Rougelsum | Gen Len |
|:-------------:|:-----:|:----:|:---------------:|:-------:|:-------:|:-------:|:---------:|:-------:|
| 0.6755        | 1.0   | 100  | 0.5339          | 60.9402 | 39.3331 | 54.6889 | 59.45     | 36.75   |
| 0.3666        | 2.0   | 200  | 0.3331          | 65.5149 | 43.0245 | 59.3121 | 62.7329   | 52.82   |
| 0.2983        | 3.0   | 300  | 0.3075          | 66.3702 | 45.5642 | 61.401  | 64.0587   | 49.97   |


### Framework versions

- Transformers 4.33.2
- Pytorch 2.4.1+cu124
- Datasets 2.20.0
- Tokenizers 0.13.3