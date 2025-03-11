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

"""
# Training the model
```sh
python speech_to_text_aed.py \
    # (Optional: --config-path=<path to dir of configs> --config-name=<name of config without .yaml>) \
    model.train_ds.tarred_audio_filepaths=<path to tar files with audio> \
    model.train_ds.manifest_filepath=<path to audio data manifest> \
    model.train_ds.batch_duration=360 \
    model.train_ds.num_buckets=30 \
    model.train_ds.bucket_duration_bins=<optional list of precomputed float bins for bucket durations, speeds up init> \
    model.validation_ds.manifest_filepath=<path to validation manifest> \
    model.test_ds.manifest_filepath=<path to test manifest> \
    model.model_defaults.asr_enc_hidden=1024 \
    model.model_defaults.lm_enc_hidden=512 \
    model.model_defaults.lm_dec_hidden=1024 \
    model.tokenizer.langs.spl_tokens.dir=<path to the directory of prompt special tokens tokenizer> \
    model.tokenizer.langs.spl_tokens.type=bpe \
    model.tokenizer.langs.en.dir=<path to the directory of en language tokenizer (add new langs the same way)> \
    model.tokenizer.langs.en.type=bpe \
    model.prompt_format="canary" \
    trainer.devices=-1 \
    trainer.accelerator="ddp" \
    trainer.max_steps=100000 \
    +trainer.limit_train_batches=20000 \
    trainer.val_check_interval=5000 \
    +trainer.use_distributed_sampler=false \
    model.optim.name="adamw" \
    model.optim.lr=0.001 \
    model.optim.betas=[0.9,0.999] \
    model.optim.weight_decay=0.0001 \
    model.optim.sched.warmup_steps=2000 \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name="<Name of experiment>" \
    exp_manager.wandb_logger_kwargs.project="<Name of project>"
```


"""
import time
import lightning.pytorch as pl
from omegaconf import OmegaConf
import sys
# print(sys.path)
sys.path.insert(0, "/share/nas169/jerryyang/NeMo")
from nemo.collections.asr.models import EncDecMultiTaskModel
from nemo.core.config import hydra_runner
from nemo.utils import logging, model_utils
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg


def get_base_model(trainer, cfg):
    """
    Returns the base model to be fine-tuned.
    Currently supports two types of initializations:
    1) `init_from_nemo_model`, and
    2) `init_from_pretrained_model`.
    Args:
        trainer: PyTorch Lightning Trainer
        cfg: config
    Returns:
        aed_model: EncDecMultiTaskModel instance
    """
    aed_model = None
    nemo_model_path = cfg.get('init_from_nemo_model', None)
    pretrained_name = cfg.get('init_from_pretrained_model', None)
    if nemo_model_path is not None and pretrained_name is not None:
        raise ValueError("Only pass `init_from_nemo_model` or `init_from_pretrained_model` but not both")
    elif nemo_model_path is None and pretrained_name is None:
        raise ValueError(
            "Both `init_from_nemo_model` and `init_from_pretrained_model cannot be None, should pass atleast one of them"
        )
    elif nemo_model_path is not None:
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

    aed_model.set_trainer(trainer)
    return aed_model

def check_vocabulary(aed_model, cfg):
    """
    Checks if the decoder and vocabulary of the model needs to be updated.
    If either of them needs to be updated, it updates them and returns the updated model.
    else vocabulary will be reused from the pre-trained model.
    Args:
        aed_model: EncDecMultiTaskModel instance
        cfg: config
    Returns:
        aed_model: EncDecMultiTaskModel instance with updated decoder and vocabulary
    """
    if hasattr(cfg.model.tokenizer, 'update_tokenizer') and cfg.model.tokenizer.update_tokenizer:
        if hasattr(cfg.model.char_labels, 'update_labels') and cfg.model.char_labels.update_labels:
            raise ValueError(
                "Both `model.tokenizer.update_tokenizer` and `model.char_labels.update_labels` cannot be passed together"
            )
        else:
            aed_model = update_tokenizer(aed_model, cfg.model.tokenizer.dir, cfg.model.tokenizer.type)
    elif hasattr(cfg.model, 'char_labels') and cfg.model.char_labels.update_labels:
        aed_model.change_vocabulary(new_vocabulary=cfg.model.char_labels.labels)
        logging.warning("The vocabulary of the model has been updated with provided char labels.")
    else:
        logging.info("Reusing the vocabulary from the pre-trained model.")

    return aed_model


def update_tokenizer(aed_model, tokenizer_dir, tokenizer_type):
    """
    Updates the tokenizer of the model and also reinitializes the decoder if the vocabulary size
    of the new tokenizer differs from that of the loaded model.
    Args:
        aed_model: EncDecMultiTaskModel instance
        tokenizer_dir: tokenizer directory
        tokenizer_type: tokenizer type
    Returns:
        aed_model: EncDecMultiTaskModel instance with updated tokenizer and decoder
    """
    vocab_size = aed_model.tokenizer.vocab_size
    transf_decoder = aed_model.transf_decoder.state_dict()
    if hasattr(aed_model, 'joint'):
        joint_state = aed_model.joint.state_dict()
    else:
        joint_state = None

    if tokenizer_dir is None:
        raise ValueError("dir must be specified if update_tokenizer is True")
    logging.info("Using the tokenizer provided through config")
    aed_model.change_vocabulary(new_tokenizer_dir=tokenizer_dir, new_tokenizer_type=tokenizer_type)
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
    """
    Sets up the training, validation and test dataloaders for the model.
    Args:
        aed_model: EncDecMultiTaskModel instance
        cfg: config
    Returns:
        aed_model: EncDecMultiTaskModel instance with updated dataloaders
    """
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

    # aed_model = EncDecMultiTaskModel(cfg=cfg.model, trainer=trainer)
    aed_model = get_base_model(trainer, cfg)
    
    # Check vocabulary type and update if needed
    aed_model = check_vocabulary(aed_model, cfg)
    
    # Setup Data
    aed_model = setup_dataloaders(aed_model, cfg)

    # Initialize the weights of the model from another model, if provided via config
    # aed_model.maybe_init_from_pretrained_checkpoint(cfg)
    trainer.fit(aed_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        if aed_model.prepare_test(trainer):
            trainer.test(aed_model)


if __name__ == '__main__':
    main()
