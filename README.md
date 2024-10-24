# Finetune for Flower Evolver website

This repo is for finetuning [nlpconnect/vit-gpt2-image-captioning](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning) with the [FlowerEvolver-dataset](https://huggingface.co/datasets/cristianglezm/FlowerEvolver-Dataset)

You can use the jupyter-notebook or the FlowerCaptioner.py script

## Download the FlowerEvolver dataset

```bash
git clone https://huggingface.co/datasets/cristianglezm/FlowerEvolver-Dataset "data"
```

## Use with transfomers

```python
from transformers import pipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FlowerCaptioner = pipeline("image-to-text", model="cristianglezm/ViT-GPT2-FlowerCaptioner", device=device)
FlowerCaptioner(["flower1.png"]) 
# A flower with 12 petals in a smooth gradient of green and blue. 
# The center is green with black accents. The stem is long and green.
```

## Install requirements

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Inference

```bash
python FlowerCaptioner.py -i <flower.png> or <folder with flowers.png>
```

## Train
 
```bash
python FlowerCaptioner.py -t -m <model_name>
```

## Convert to ONNX

````bash
python convert.py --quantize --model_id "./models/FlowerCaptioner" --task "image-to-text-with-past" --opset 18
````

## License

convert.py is under [huggingface/transformers.js](https://github.com/huggingface/transformers.js) license.

quantize.py is under [huggingface/transformers.js](https://github.com/huggingface/transformers.js) license.

