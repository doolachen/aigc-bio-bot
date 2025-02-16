# aigc-bio-bot
This repo is used to create a bio-medical bot. Deepseek-R1 model is used.

## Requirements
`pip install -r requirements.txt`

## Step
### 1.  Download DeepSeek Model
`export PYTHONPATH=<PATH_TO_PROJECT>`

`python models/download.py`

We will get `DeepSeek-R1-Distill-Qwen-32B` in your models folder.

### 2. Prepare Dataset

`python dataset/gen_dataset.py`

We will get `medical-o1-reasoning-dataset` in your dataset folder.