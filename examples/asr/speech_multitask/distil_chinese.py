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
# python distil_chinese.py --config-name=distil_matbn_2_eng

import time
import lightning.pytorch as pl
from omegaconf import OmegaConf
import copy
import os
import sys
sys.path.insert(0, "/share/nas169/jerryyang/NeMo")
from nemo.collections.asr.models import EncDecMultiTaskModel
from nemo.core.config import hydra_runner
from nemo.utils import logging, model_utils
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

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

def partial_init_student_from_teacher(teacher_model, student_model):

    # 1) 複製 encoder
    student_model.encoder.load_state_dict(
        teacher_model.encoder.state_dict(), 
        strict=True
    )

    # 2) 複製 transf_decoder
    student_model.transf_decoder.load_state_dict(
        teacher_model.transf_decoder.state_dict(),
        strict=False
    )

class DistillationWrapper:
    def __init__(self, student: EncDecMultiTaskModel, teacher: EncDecMultiTaskModel):
        self.student = student
        self.teacher = teacher
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    def training_step(self, batch, batch_idx):
        return self.student.training_step_with_teacher(batch, batch_idx, self.teacher)

    def validation_step(self, batch, batch_idx):
        return self.student.validation_step_with_teacher(batch, batch_idx, self.teacher)

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
    
    teacher_model = EncDecMultiTaskModel(cfg=cfg.model, trainer=trainer)
    teacher_model.maybe_init_from_pretrained_checkpoint(cfg)

    cfg_student = copy.deepcopy(cfg)
    OmegaConf.set_struct(cfg_student, False)
    if "init_from_ptl_ckpt" in cfg_student:
        del cfg_student["init_from_ptl_ckpt"]
    cfg_student["init_from_pretrained_model"] = "nvidia/canary-180m-flash"
    cfg_student["model"]["use_bert"] = False
    cfg_student["model"]["transf_decoder"]["config_dict"]["add_gated_x_attn"] = False

    student_model = EncDecMultiTaskModel(cfg=cfg_student.model, trainer=trainer)
    partial_init_student_from_teacher(teacher_model, student_model)
    student_model.set_trainer(trainer)
    
    # Check vocabulary type and update if needed
    teacher_model = check_vocabulary(teacher_model, cfg)
    student_model = check_vocabulary(teacher_model, cfg_student)

    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False

    # Freeze all
    for name, param in student_model.named_parameters():
        param.requires_grad = False

    # Unfreeze decoder
    for name, param in student_model.transf_decoder.named_parameters():
        param.requires_grad = True

    # Unfreeze output head
    for name, param in student_model.log_softmax.named_parameters():
        param.requires_grad = True

    # Avoid key error
    teacher_model.change_prompt()
    student_model.change_prompt()

    # Setup Data
    teacher_model = setup_dataloaders(teacher_model, cfg)
    student_model = setup_dataloaders(student_model, cfg_student)
    
    # Setup Optimizer
    student_model.setup_optimization(cfg.model.optim)

    # Setup SpecAug
    if hasattr(cfg.model, 'spec_augment') and cfg.model.spec_augment is not None:
        student_model.spec_augment = EncDecMultiTaskModel.from_config_dict(cfg.model.spec_augment)

    # trainer.fit(student_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        if student_model.prepare_test(trainer):
            trainer.test(student_model)

if __name__ == '__main__':
    main()
