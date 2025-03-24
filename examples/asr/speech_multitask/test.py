import glob
import json
import librosa
import numpy as np
from omegaconf import OmegaConf, open_dict
import os
import soundfile as sf
import subprocess
import tarfile
import tqdm
import wget

import torch
import sys
sys.path.insert(0, "/share/nas169/jerryyang/NeMo")

from nemo.collections.asr.models import EncDecMultiTaskModel

map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
canary_model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-180m-flash', map_location=map_location)

base_model_cfg = OmegaConf.load("../conf/speech_multitask/base.yaml")

base_model_cfg['name'] = 'canary-180m-flash-finetune'
base_model_cfg.pop("init_from_nemo_model", None)
base_model_cfg['init_from_pretrained_model'] = "nvidia/canary-180m-flash"

canary_model.save_tokenizers('./canary_flash_tokenizers/')

for lang in os.listdir('canary_flash_tokenizers'):
    base_model_cfg['model']['tokenizer']['langs'][lang] = {}
    base_model_cfg['model']['tokenizer']['langs'][lang]['dir'] = os.path.join('canary_flash_tokenizers', lang)
    base_model_cfg['model']['tokenizer']['langs'][lang]['type'] = 'bpe'
base_model_cfg['spl_tokens']['model_dir'] = os.path.join('canary_flash_tokenizers', "spl_tokens")

base_model_cfg['model']['prompt_format'] = canary_model._cfg['prompt_format']
base_model_cfg['model']['prompt_defaults'] = canary_model._cfg['prompt_defaults']

base_model_cfg['model']['model_defaults'] = canary_model._cfg['model_defaults']
base_model_cfg['model']['preprocessor'] = canary_model._cfg['preprocessor']
base_model_cfg['model']['encoder'] = canary_model._cfg['encoder']
base_model_cfg['model']['transf_decoder'] = canary_model._cfg['transf_decoder']
base_model_cfg['model']['transf_encoder'] = canary_model._cfg['transf_encoder']
     
cfg = OmegaConf.create(base_model_cfg)
with open("../conf/speech_multitask/canary-180m-flash-finetune.yaml", "w") as f:
    OmegaConf.save(cfg, f)

# MANIFEST = os.path.join("datasets", "LibriLight", 'train_manifest.json')
# python test.py \
#   --config-path="../conf/speech_multitask" \
#   --config-name="canary-180m-flash-finetune.yaml" \
#   name="canary-180m-flash-finetune" \
#   model.train_ds.manifest_filepath={/datasets/LibriLight/train_manifest.json} \
#   model.validation_ds.manifest_filepath={/datasets/LibriLight/train_manifest.json} \
#   model.test_ds.manifest_filepath={/datasets/LibriLight/train_manifest.json} \
#   exp_manager.exp_dir="canary_results" \
#   exp_manager.resume_ignore_no_checkpoint=true \
#   trainer.max_steps=10 \
#   trainer.log_every_n_steps
