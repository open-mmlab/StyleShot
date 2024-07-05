# Tutorial: Prepare datasets

StyleGallery includes <a href='https://github.com/JourneyDB/JourneyDB'>JourneyDB<a>, a dataset comprising a broad spectrum of diverse styles derived from MidJourney, and <a href='https://huggingface.co/datasets/huggan/wikiart'>WIKIART<a>, with extensive fine-grained painting styles, such as pointillism and ink drawing, and a subset of stylized images from <a href='https://console.cloud.google.com/storage/browser/sfr-unicontrol-data-research/dataset;tab=objects?pli=1&prefix=&forceOnObjectsSortingFiltering=false'>MultiGen-20M<a> ( A subset of LAION-Aesthetics).

If your folder structure is different, you may need to change the corresponding paths in config files.

```none
StyleShot
├── annotator
├── assets
├── ip_adapter
├── json_files
│   ├── stylegallery.jsonl
├── data
│   ├── JourneyDB
│   │   ├── data
│   │   │   ├── train
│   │   │   │   ├── train_anno_realease_repath.jsonl
│   │   │   │   ├── imgs
│   ├── wikiart
│   │   ├── images
│   │   ├── dataset_infos.json
│   │   ├── data.json
│   ├── MultiGen-20M
│   │   ├── images
│   │   ├── json_files
│   │   │   ├── aesthetics_plus_all_group_canny_all.json

```

## Download dataset

StyleGallery is constructed by three open source datasets. Before proceeding with further processing, you need to download these three datasets  <a href='https://github.com/JourneyDB/JourneyDB'>JourneyDB<a>, <a href='https://huggingface.co/datasets/huggan/wikiart'>WIKIART<a> and  <a href='https://console.cloud.google.com/storage/browser/sfr-unicontrol-data-research/dataset;tab=objects?pli=1&prefix=&forceOnObjectsSortingFiltering=false'>MultiGen-20M<a> according to the guidance provided in the corresponding links.

## WIKIART

The images on WIKIART lack corresponding caption descriptions; therefore, they must be labeled prior to use.

We develop a simple script to generate captions for images in WikiArt using the <a href='https://huggingface.co/docs/transformers/model_doc/blip-2'>BLIP2</a> model:

```shell
import os
import sys
import json
import glob
import torch
from PIL import Image
from lavis.models import load_model_and_preprocess

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

root = "./data/wikiart/"
jsonf = "./data/wikiart/data.json"
filenames = [os.path.join(root, "images", line['path']) for line in json.load(open(jsonf))]
savepaths = [os.path.join(root, "blipcaptions", line['path'][:-4]+".txt") for line in json.load(open(jsonf))]

model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device)

for idx in range(len(filenames)):
    filename = filenames[idx]
    savepath = savepaths[idx]

    name = filename.split('/')[-1].split('.')[0]

    raw_image = Image.open(filename).convert("RGB")
    image = raw_image.resize((512, 512))
    
    image_pt = vis_processors["eval"](image).unsqueeze(0).to(device)

    results = model.generate({"image": image_pt, "prompt": "Question: Describe this image in detail. Answer: "}, max_length=77)[0]

    with open(savepath, 'w') as fid:
        fid.write(results+'\n')
```


## MultiGen-20M

To enhance the diversity of styles in StyleGallery, we filter the style data from MultiGen-20M using the following script.

Initially, we created a style descriptor banks that incorporates descriptors from ./data/wikiart/dataset_infos.json, manually added descriptors and regular expressions for years (This was based on our observation that the inclusion of a year in MultiGen captions typically signifies a specific style associated with that period).

```shell
import json
with open('./data/wikiart/dataset_infos.json', 'r') as f:
    dataset_infos = f.readlines()
dataset_infos = json.loads(dataset_infos[0])
Artist = dataset_infos['huggan--wikiart']['features']['artist']['names']
Genre = dataset_infos['huggan--wikiart']['features']['genre']['names']
Style = dataset_infos['huggan--wikiart']['features']['style']['names']
Banks = Style + Genre + Artist + ['watercolor', 'painting', 'paint', 'art', 'draw', 'drawing', 'abstract', 'neoclass', 'rococo', 'monet', 'cubism', 'portrait', 'engrave', 'ink draw']
```

Subsequently, we review the captions of each image in MultiGen to identify any style descriptors from our banks. Upon detecting a descriptor, the image is classified as a style image and documented in a newly created JSON file:

```shell
import re
with open('./data/MultiGen-20M/json_files/aesthetics_plus_all_group_canny_all.json', 'r') as f:
    datas = f.readlines()
for data in datas:
    tmp = json.loads(data)
    caption = tmp['prompt']
    if caption is None:
        caption = ""
    for b in Banks:
        if b in caption:
            caption = caption.replace(b, "")
            flag = True
    pattern1 = r'\d{4}'
    pattern2 = r'\d{3}'

    # years
    if re.search(pattern1, caption) or re.search(pattern2, caption):
        flag = True
    tmp['content_prompt'] = caption
    if flag:
        with open('./json_files/aesthetics_plus_all_group_canny_all_content.jsonl', 'a') as f:
            j = json.dumps(tmp)
            f.write(j)
            f.write("\n")
        flag = False
```

Finally, the style subset of Multi-Gen 20M includes 0.87M style images.

## De-stylization

To improve the learning of style embeddings from StyleGallery, we endeavor to remove all style-related descriptions from the text across all text-image pairs, retaining only content-related text.

### Style subset of MultiGen-20M
During the MultiGen filtering process, we have achieved de-stylization by replacing the style descriptors in the captions with an empty string "".

### WIKIART
For WIKIART, we also employ style descriptor banks to achieve de-stylization by replacing the style descriptors in the captions with an empty string "":

```shell
import json

with open('./data/wikiart/dataset_infos.json', 'r') as f:
    dataset_infos = f.readlines()
dataset_infos = json.loads(dataset_infos[0])
Artist = dataset_infos['huggan--wikiart']['features']['artist']['names']
Genre = dataset_infos['huggan--wikiart']['features']['genre']['names']
Style = dataset_infos['huggan--wikiart']['features']['style']['names']
Style.append('Ink')
Banks = Style + Genre + Artist + ['watercolor', 'painting', 'colorful', 'express', 'real', 'Impress', 'romantic', 'surreal', 'symbol', 'abstract', 'neoclass', 'rococo', 'monet', 'cubism', 'academic', 'pop', 'portrait', 'landscape', 'drawing', 'religious', 'cityscape', 'engrav', 'illustra', 'photo', 'poster', 'ink']

datas = json.load(open('./data/wikiart/data.json'))

# oversampling nine times
files = ['images/', 'images/', 'images/', 'images/', 'images/', 'images/', 'images/', 'images/', 'images/']

for data in datas:
    tmp = json.loads(data)
    with open(os.path.join('./data/wikiart/blipcaption/', tmp['path'][:-4]+".txt")) as f:
        caption = f.readline()[:-1]
    for b in Banks:
        if b in caption:
            caption = caption.replace(b, "")
    tmp['prompt'] = caption
    with open('./json_files/wikiart_with_caption.jsonl', 'a') as f:
        for file in files:
            t = tmp.copy()
            t['image_file'] = file + t['path']
            j = json.dumps(t)
            f.write(j)
            f.write("\n")
```

We found that WIKIART contains only 80k images. To balance the style distribution within the dataset, we oversampled WikiArt to an augmented dataset comprising 0.73M images.

### JourneyDB

In JourneyDB, style descriptors are explicitly marked. Consequently, we simply replace the style descriptors in the captions with an empty string "".

```shell
import json

def get_content_prompt(data):
    try:
        prompt = data['prompt']
    except:
        prompt = ""
    try:
        styles = data['Task1']['Style']
    except:
        styles = []
    for s in styles:
        prompt = prompt.replace(s, "")
    return prompt

with open("./data/JourneyDB/data/train/train_anno_realease_repath.jsonl", 'r', encoding='utf-8') as f:
    datas = f.readlines()

for data in datas:
    data = json.loads(data)
    content_prompt = get_content_prompt(data)
    data['content_prompt'] = content_prompt
    with open("./json_files/train_anno_realease_repath_content_prompt.jsonl", 'a') as f:
        j = json.dumps(data)
        f.write(j)
        f.write("\n")
```


## Process content input

To train our content-fusion encoder, we process content inputs using the <a href='https://github.com/s9xie/hed'>HED</a> detector, with additional thresholding and dilation steps, implemented in the [./annotator/](./annotator/) . Use the following script to process the content input:


```shell
import cv2
from annotator.hed import SOFT_HEDdetector

img_path = "content image.png"

detector = SOFT_HEDdetector()
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
content_input = detector(img)
```

## Json file download

We have uploaded our JSON files to the following <a href='https://drive.google.com/drive/folders/10T3t58rQKDmYOLschUYj0tzm6zuOngMd?usp=drive_link'>URL</a>. You can directly download them.
Our JSON file, stylegallery.jsonl, consolidates JSON files from JourneyDB, MultiGen-20M, and WikiArt into a uniform format:

```shell
{"image_file": "", "content_prompt": "", ...}
{"image_file": "", "content_prompt": "", ...}
{"image_file": "", "content_prompt": "", ...}
...
```