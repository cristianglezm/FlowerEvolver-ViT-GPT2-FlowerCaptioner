import json
import os

class FlowerDataset:
    def __init__(self, json_file, root_dir):
        self.root_dir = root_dir

        with open(json_file, 'r') as f:
            data = json.load(f)
            self.images = [item["ImagePath"] for item in data['annotations']]
            self.captions = [item["caption"] for item in data['annotations']]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        caption = self.captions[idx]
        img_full_path = f"{self.root_dir}/{img_path}"
        return {
            'image_path': img_full_path,
            'caption': caption
        }
    def generator(self):
        for idx in range(len(self)):
            yield self[idx]
