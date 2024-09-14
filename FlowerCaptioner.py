#!/bin/python

from transformers import VisionEncoderDecoderModel, GenerationConfig, default_data_collator, pipeline
from transformers import ViTImageProcessor, GPT2TokenizerFast, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset, DatasetDict 
from PIL import Image
from pathlib import Path
from textwrap import wrap
import numpy as np
import matplotlib.pyplot as plt
import evaluate
import torch
import json
import nltk
import os
import random
import argparse
import sys
import FlowerDataset

def finetune(modelFolder, device, epoches=3, batchSize=4):
    """
    Fine-tunes a VisionEncoderDecoder model for image captioning on a flowerEvolver dataset.

    Args:
        modelFolder (str): Directory to save the fine-tuned model.
        device (str): Device for training ('cuda' or 'cpu').
        epoches (int, optional): Number of training epochs. Defaults to 3.
        batchSize (int, optional): Training batch size. Defaults to 4.

    Returns:
        None

    Creates:
        - Saves the fine-tuned model, tokenizer, and feature extractor to `modelFolder`.
        - Creates a model card for the fine-tuned model.

    """
    try:
       nltk.data.find("tokenizers/punkt")
       nltk.data.find("punkt_tab")
    except (LookupError, OSError):
        nltk.download("punkt", quiet=True)
        nltk.download('punkt_tab', quiet=True)
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning", clean_up_tokenization_spaces=True)

    # GPT2 only has bos/eos tokens but not decoder_start/pad tokens
    tokenizer.pad_token = tokenizer.eos_token
    # update the model config
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.generation_config = GenerationConfig(
        eos_token_id = tokenizer.eos_token_id,
        decoder_start_token_id = tokenizer.bos_token_id,
        pad_token_id = tokenizer.pad_token_id,
        max_new_tokens = 128
    )
    # text preprocessing step
    def tokenization_fn(captions, max_target_length):
        return tokenizer(captions, padding="max_length", max_length=max_target_length).input_ids


    # image preprocessing step
    def feature_extraction_fn(image_paths):
        images = [Image.open(image_file).convert("RGB") for image_file in image_paths]
        encoder_inputs = feature_extractor(images=images, return_tensors="np")
        return encoder_inputs.pixel_values


    def preprocess_fn(dataset, max_target_length):
        image_paths = dataset['image_path']
        captions = dataset['caption']

        model_inputs = {}
        model_inputs['labels'] = tokenization_fn(captions, max_target_length)
        model_inputs['pixel_values'] = feature_extraction_fn(image_paths)

        return model_inputs


    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels


    ignore_pad_token_for_loss = True
    metric = evaluate.load("rouge")


    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        preds = preds.astype(np.int64)
        labels = labels.astype(np.int64)
        if isinstance(preds, tuple):
            preds = preds[0]
        if ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)


        decoded_preds, decoded_labels = postprocess_text(decoded_preds,
                                                         decoded_labels)

        result = metric.compute(predictions=decoded_preds,
                                references=decoded_labels,
                                use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result
    data = Path("./data")
    if(not data.resolve().exists()):
        print("please use:\ngit clone https://huggingface.co/datasets/cristianglezm/FlowerEvolver-Dataset \"data\"\n to download the Flower dataset")
    else:
        print("flowers dataset found.")
    # Create the custom dataset
    flower_dataset = FlowerDataset.FlowerDataset(
        json_file="./data/captions.json",
        root_dir="./data"
    )

    # Convert the custom dataset to a Hugging Face Dataset
    hf_dataset = Dataset.from_generator(flower_dataset.generator)

    # Split the dataset into training and validation sets
    dataset = hf_dataset.train_test_split(test_size=0.2)
    processed_dataset = dataset.map(
        function = preprocess_fn,
        batched = True,
        fn_kwargs = {"max_target_length": model.generation_config.max_new_tokens},
        remove_columns = dataset['train'].column_names
    )
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="epoch", # change to eval_strategy="epoch",
        num_train_epochs=epoches,
        per_device_train_batch_size=batchSize,
        per_device_eval_batch_size=batchSize,
        warmup_steps=500,
        weight_decay=0.01,
        #logging_dir='./logs',
        logging_steps=10,
        output_dir="./FlowerCaptioner-training",
        report_to=None
    )
    device = torch.device(device)
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=feature_extractor,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=processed_dataset['train'],
        eval_dataset=processed_dataset['test'],
        data_collator=default_data_collator,
    )
    print("training model")
    trainer.train()
    print("saving model...")
    output_dir = modelFolder
    trainer.save_model(output_dir)
    trainer.create_model_card(language="english", license="apache 2.0",
                              tags=["art"],
                              model_name="ViT-GPT2-FlowerCaptioner",
                              finetuned_from="nlpconnect/vit-gpt2-image-captioning",
                              tasks="image-to-text",
                              dataset_tags=["art"],
                              dataset="cristianglezm/FlowerEvolver-dataset")
    tokenizer.save_pretrained(output_dir)
    feature_extractor.save_pretrained(output_dir)


def describe(modelFolder, device, flowersImages):
    """
    Generate captions for images in a directory and save them in a JSON file or print the caption if given a single image file.

    Args:
        modelFolder (str): Path to the model directory or model identifier.
        device (str): Device to use ('cuda' for GPU or 'cpu' for CPU).
        flowersImages (str): Path to a directory with images or a single image file.

    Returns:
        None

    Creates:
        A JSON file named `foldername.json` with annotations if `flowersImages` is a directory.
        The JSON contains a list of objects with `ImagePath` and `caption`.

    """
    flowersToDescribe = []
    isFolder = False
    if os.path.isfile(flowersImages):
        flowersToDescribe.append(flowersImages)
    elif os.path.isdir(flowersImages):
        isFolder = True
        for file in os.listdir(flowersImages):
            if file.endswith(".png"):
                flowersToDescribe.append(os.path.join(flowersImages, file))
    modelName = modelFolder
    if not os.path.exists(modelFolder):
        print(f"model {modelFolder} does not exists")
        print("downloading model from cristianglezm/ViT-GPT2-FlowerCaptioner")
        modelName = "cristianglezm/ViT-GPT2-FlowerCaptioner"
    FlowerCaptioner = pipeline("image-to-text", model=modelName, device=device)
    captions = [captions[0]["generated_text"] for captions in FlowerCaptioner(flowersToDescribe)]
    annotations = []
    captionsFilename = os.path.split(os.path.normpath(flowersImages))[1]
    for i in range(0, len(flowersToDescribe)):
        annotations.append({
            "ImagePath": f"{captionsFilename}/{os.path.split(flowersToDescribe[i])[1]}",
            "caption": captions[i]
        })
        print(f"{annotations[i]['ImagePath']} has the following description:\n {captions[i]}")
    if isFolder:
        print(f"writing {captionsFilename}.json file")
        with open(f"{captionsFilename}.json", 'w', encoding='utf-8') as f:
            json.dump({
                "annotations": annotations
            }, f, ensure_ascii=False, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        action="store_true",
        help="finetune the model"
    )
    parser.add_argument(
        "-i",
        type=str,
        metavar='flower_images',
        nargs='?',
        help="if a folder is given it will describe each flower inside it"
    )
    parser.add_argument(
        "-m",
        type=str,
        metavar='model',
        nargs='?',
        help="load or save name for the model",
        default="./models/FlowerCaptioner"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="number of epochs for training (default: 3)"
    )
    parser.add_argument(
        "--batchSize",
        type=int,
        default=4,
        help="batch size for training (default: 4)"
    )
    opt = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help(sys.stderr)
        sys.exit(1)
    modelFolder = opt.m
    flowersImages = opt.i
    epoch = opt.epochs
    batchSize = opt.batchSize
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    if opt.t:
        finetune(modelFolder, device, epoch, batchSize)
    elif opt.i:
        print(f"processing {flowersImages}")
        describe(modelFolder, device, flowersImages)
if __name__ == "__main__":
    main()
