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
- Loss: 0.4930
- Rouge1: 68.3498
- Rouge2: 46.7534
- Rougel: 62.3763
- Rougelsum: 65.9575
- Gen Len: 49.82

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
- num_epochs: 25

### Training results

| Training Loss | Epoch | Step | Validation Loss | Rouge1  | Rouge2  | Rougel  | Rougelsum | Gen Len |
|:-------------:|:-----:|:----:|:---------------:|:-------:|:-------:|:-------:|:---------:|:-------:|
| 0.6986        | 1.0   | 100  | 0.5339          | 64.9813 | 42.4686 | 58.2586 | 63.3933   | 47.25   |
| 0.3408        | 2.0   | 200  | 0.3263          | 67.5461 | 46.5219 | 62.7962 | 65.6509   | 47.39   |
| 0.2797        | 3.0   | 300  | 0.2829          | 65.0704 | 42.0682 | 58.4268 | 63.2368   | 56.8    |
| 0.2584        | 4.0   | 400  | 0.2588          | 65.5074 | 45.227  | 60.2469 | 63.4253   | 52.25   |
| 0.2589        | 5.0   | 500  | 0.2607          | 66.7346 | 45.8264 | 61.7373 | 64.8857   | 50.64   |
| 0.2179        | 6.0   | 600  | 0.2697          | 63.8334 | 42.997  | 58.1585 | 61.7704   | 52.43   |
| 0.1662        | 7.0   | 700  | 0.2631          | 68.6188 | 48.3329 | 63.9474 | 66.6006   | 46.94   |
| 0.161         | 8.0   | 800  | 0.2749          | 69.0046 | 48.1421 | 63.7844 | 66.8317   | 49.74   |
| 0.1207        | 9.0   | 900  | 0.3117          | 70.0357 | 48.9002 | 64.416  | 67.7582   | 48.66   |
| 0.0909        | 10.0  | 1000 | 0.3408          | 65.9578 | 45.2324 | 60.2838 | 63.7493   | 46.92   |
| 0.0749        | 11.0  | 1100 | 0.3516          | 67.4244 | 46.1985 | 61.6408 | 65.5371   | 46.61   |
| 0.0665        | 12.0  | 1200 | 0.3730          | 68.6911 | 47.7089 | 63.0381 | 66.6956   | 47.89   |
| 0.0522        | 13.0  | 1300 | 0.3891          | 67.2365 | 45.4165 | 61.4063 | 64.857    | 48.91   |
| 0.0355        | 14.0  | 1400 | 0.4128          | 69.1494 | 47.9278 | 63.3334 | 66.5969   | 50.55   |
| 0.0309        | 15.0  | 1500 | 0.4221          | 66.2447 | 44.937  | 60.1403 | 63.8541   | 50.71   |
| 0.0265        | 16.0  | 1600 | 0.4343          | 67.8178 | 46.7084 | 61.8173 | 65.4375   | 50.85   |
| 0.0158        | 17.0  | 1700 | 0.4577          | 67.9846 | 45.9562 | 61.6353 | 65.7207   | 50.81   |
| 0.0166        | 18.0  | 1800 | 0.4731          | 69.0971 | 47.7001 | 62.856  | 66.7796   | 50.01   |
| 0.0121        | 19.0  | 1900 | 0.4657          | 68.1397 | 46.4258 | 62.2696 | 65.9332   | 49.15   |
| 0.0095        | 20.0  | 2000 | 0.4793          | 68.6497 | 47.9446 | 63.0466 | 66.5409   | 50.96   |
| 0.0086        | 21.0  | 2100 | 0.4780          | 68.4363 | 46.7296 | 62.359  | 66.2626   | 50.02   |
| 0.0068        | 22.0  | 2200 | 0.4863          | 67.5415 | 46.0821 | 61.57   | 65.4613   | 49.5    |
| 0.0061        | 23.0  | 2300 | 0.4892          | 68.1283 | 46.5802 | 62.0832 | 66.0203   | 50.21   |
| 0.006         | 24.0  | 2400 | 0.4912          | 68.1723 | 46.3239 | 62.2007 | 65.6725   | 49.89   |
| 0.0057        | 25.0  | 2500 | 0.4930          | 68.3498 | 46.7534 | 62.3763 | 65.9575   | 49.82   |


### Framework versions

- Transformers 4.43.4
- Pytorch 2.4.1+cu124
- Datasets 2.20.0
- Tokenizers 0.19.1
