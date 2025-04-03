# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# my command
# python chinese.py --config-name=matbn_2

import time
import lightning.pytorch as pl
from omegaconf import OmegaConf
import os
import sys
sys.path.insert(0, "/share/nas169/jerryyang/NeMo")
from nemo.collections.asr.models import EncDecMultiTaskModel
from nemo.core.config import hydra_runner
from nemo.utils import logging, model_utils
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg


def get_base_model(trainer, cfg):
    aed_model = None
    nemo_model_path = cfg.get('init_from_nemo_model', None)
    pretrained_name = cfg.get('init_from_pretrained_model', None)
    ptl_ckpt_path = cfg.get('init_from_ptl_ckpt', None)

    # Ensure that only one initialization option is provided
    provided_options = [opt for opt in [nemo_model_path, pretrained_name, ptl_ckpt_path] if opt is not None]
    if len(provided_options) > 1:
        raise ValueError("Only one initialization parameter can be provided: init_from_nemo_model, init_from_pretrained_model, or init_from_ptl_ckpt")
    elif len(provided_options) == 0:
        raise ValueError("At least one initialization parameter must be provided: init_from_nemo_model, init_from_pretrained_model, or init_from_ptl_ckpt")
    
    if nemo_model_path is not None:
        # Restore model from Nemo model file
        aed_model = EncDecMultiTaskModel.restore_from(restore_path=nemo_model_path)
    elif pretrained_name is not None:
        # Due to potential first time download of the model on the cluster, we need to make sure that only one
        # rank downloads the model and the others wait for the download to finish.
        num_ranks = trainer.num_devices * trainer.num_devices

        if num_ranks > 1 and is_global_rank_zero():
            aed_model = EncDecMultiTaskModel.from_pretrained(model_name=pretrained_name)
        else:
            # Sleep on all ranks for at least 60 seconds
            wait_time = int(cfg.get('exp_manager', {}).get('seconds_to_sleep', 60))
            if wait_time < 60:
                wait_time = 60

            logging.info(f"Sleeping for at least {wait_time} seconds to wait for model download to finish.")

            time.sleep(wait_time)

            # restore model from cached model dir
            aed_model = EncDecMultiTaskModel.from_pretrained(model_name=pretrained_name)
    elif ptl_ckpt_path is not None:
        aed_model = EncDecMultiTaskModel(cfg=cfg.model, trainer=trainer)
        aed_model.maybe_init_from_pretrained_checkpoint(cfg)

    aed_model.set_trainer(trainer)
    return aed_model

def check_vocabulary(aed_model, cfg):
    if hasattr(cfg.model.tokenizer, 'update_tokenizer') and cfg.model.tokenizer.update_tokenizer:
        if hasattr(cfg.model.char_labels, 'update_labels') and cfg.model.char_labels.update_labels:
            raise ValueError(
                "Both `model.tokenizer.update_tokenizer` and `model.char_labels.update_labels` cannot be passed together"
            )
        else:
            aed_model = update_tokenizer(aed_model, cfg.model.tokenizer)
    elif hasattr(cfg.model, 'char_labels') and cfg.model.char_labels.update_labels:
        aed_model.change_vocabulary(new_vocabulary=cfg.model.char_labels.labels)
        logging.warning("The vocabulary of the model has been updated with provided char labels.")
    else:
        logging.info("Reusing the vocabulary from the pre-trained model.")

    return aed_model


def update_tokenizer(aed_model, tokenizer_cfg):
    vocab_size = aed_model.tokenizer.vocab_size
    transf_decoder = aed_model.transf_decoder.state_dict()
    if hasattr(aed_model, 'joint'):
        joint_state = aed_model.joint.state_dict()
    else:
        joint_state = None
    logging.info("Using the tokenizer provided through config")
    
    aed_model.change_vocabulary(new_tokenizer_dir=tokenizer_cfg, new_tokenizer_type=tokenizer_cfg.type)
    if aed_model.tokenizer.vocab_size != vocab_size:
        logging.warning(
            "The vocabulary size of the new tokenizer differs from that of the loaded model. As a result, finetuning will proceed with the new vocabulary, and the decoder will be reinitialized."
        )
    else:
        aed_model.transf_decoder.load_state_dict(transf_decoder)
        if joint_state is not None:
            aed_model.joint.load_state_dict(joint_state)

    return aed_model

def setup_dataloaders(aed_model, cfg):
    cfg = model_utils.convert_model_config_to_dict_config(cfg)
    aed_model.setup_training_data(cfg.model.train_ds)
    aed_model.setup_multiple_validation_data(cfg.model.validation_ds)
    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        aed_model.setup_multiple_test_data(cfg.model.test_ds)

    return aed_model

@hydra_runner(config_path="../conf/speech_multitask/", config_name="fast-conformer_aed")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
    exp_manager(trainer, cfg.get("exp_manager", None))

    # Check for spl tokens to create spl_tokenizer.
    if cfg.get("spl_tokens"):
        logging.info("Detected spl_tokens config. Building tokenizer.")
        spl_cfg = cfg["spl_tokens"]
        spl_tokenizer_cls = model_utils.import_class_by_path(cfg.model.tokenizer.custom_tokenizer["_target_"])
        spl_tokenizer_cls.build_special_tokenizer(
            spl_cfg["tokens"], spl_cfg["model_dir"], force_rebuild=spl_cfg["force_rebuild"]
        )
        cfg.model.tokenizer.langs.spl_tokens.dir = spl_cfg["model_dir"]
    
    aed_model = get_base_model(trainer, cfg)

    # Check vocabulary type and update if needed
    aed_model = check_vocabulary(aed_model, cfg)

    # Avoid key error
    aed_model.change_prompt()

    # Setup Data
    aed_model = setup_dataloaders(aed_model, cfg)
    
    # Setup Optimizer
    aed_model.setup_optimization(cfg.model.optim)

    # Setup SpecAug
    if hasattr(cfg.model, 'spec_augment') and cfg.model.spec_augment is not None:
        aed_model.spec_augment = EncDecMultiTaskModel.from_config_dict(cfg.model.spec_augment)

    trainer.fit(aed_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        if aed_model.prepare_test(trainer):
            trainer.test(aed_model)

if __name__ == '__main__':
    main()
