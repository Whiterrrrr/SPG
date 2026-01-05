# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import wandb
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig, TrainerCallback
from trl import TrlParser, ModelConfig
from peft import LoraConfig, PeftModel, get_peft_model

# Custom imports
from diffu_grpo_trainer import DiffuGRPOTrainer
from spg_trainer import SPGTrainer
from diffu_grpo_config import DiffuGRPOConfig
from reward_func import (
    xmlcount_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    int_reward_func,
    correctness_reward_func,
    countdown_reward_func,
    correctness_reward_func_math,
    sudoku_reward_func,
    boxed_and_answer_tags_format_reward,
    reward_len,
)
from data_utils import (
    get_gsm8k_questions,
    get_countdown_questions,
    get_sudoku_questions,
    get_sudoku_questions_new,
    set_random_seed,
    get_math_questions,
)
from train_utils import (
    find_latest_checkpoint, 
    ResumeStepCallback
)

def main(grpo_config, model_config):

    # Set seed for reproducibility
    set_random_seed(grpo_config.seed)

    # Load dataset based on configuration
    if grpo_config.dataset == "gsm8k":
        dataset = get_gsm8k_questions("train")
        reward_functions = [
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ]
    elif grpo_config.dataset == "countdown":
        dataset = get_countdown_questions("train")
        reward_functions = [countdown_reward_func]
    # elif grpo_config.dataset == "sudoku":
    #     dataset = get_sudoku_questions()
    #     reward_functions = [sudoku_reward_func]
    elif grpo_config.dataset == "sudoku_new":
        dataset = get_sudoku_questions_new(few_shot=grpo_config.few_shot)
        reward_functions = [sudoku_reward_func]
    elif grpo_config.dataset == "math":
        dataset = get_math_questions("train")
        reward_functions = [
            correctness_reward_func_math,
            boxed_and_answer_tags_format_reward,
        ]

    # Shuffle dataset with fixed seed for reproducibility
    dataset = dataset.shuffle(seed=grpo_config.seed)

    # Split dataset if needed
    if grpo_config.dataset in ["countdown", "sudoku", "sudoku_new"]:
        train_set = dataset.select(range(0, len(dataset) - 500))  # Leave last 500 for evaluation
    else:
        train_set = dataset

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4 bit quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load model and tokenizer
    model = AutoModel.from_pretrained(
        grpo_config.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(grpo_config.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False

    # Configure LoRA for parameter-efficient fine-tuning
    peft_config = LoraConfig(
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=model_config.lora_dropout,
    )

    # Try to load a PEFT checkpoint if available. If it fails or doesn't provide trainable params,
    # we will attempt to enable adapter/Lora parameters programmatically below.
    loaded_checkpoint_step = None
    # If a resume checkpoint path is provided in the config, try loading the PEFT checkpoint from it.
    if getattr(grpo_config, "resume_checkpoint_path", None):
        base_cp = grpo_config.resume_checkpoint_path
        cp_path, cp_step = find_latest_checkpoint(base_cp)
        # If no checkpoint subdir found, try using the provided path directly
        checkpoint_to_load = cp_path or base_cp
        try:
            model = PeftModel.from_pretrained(
                model,
                checkpoint_to_load,
                torch_dtype=torch.bfloat16,
            ).to(device)
            loaded_checkpoint_step = cp_step
        except Exception:
            # If loading fails, fall back to base model and continue; we'll attempt to enable adapters programmatically.
            model = model

    # Build optimizer from trainable parameters and pass it into the trainer so DeepSpeed
    # receives a valid optimizer (avoids `param_groups` empty issue).
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # If no trainable params found, attempt to enable typical PEFT/Lora parameters by name.
    if len(trainable_params) == 0:
        adapter_keywords = ("lora", "adapter", "alpha", "up_proj", "down_proj", "lora_up", "lora_down")
        enabled = False
        for name, p in model.named_parameters():
            if any(k in name.lower() for k in adapter_keywords):
                if torch.is_floating_point(p) or torch.is_complex(p):
                    p.requires_grad = True
                    enabled = True
        if enabled:
            trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(trainable_params, lr=grpo_config.learning_rate)

    if grpo_config.trainer == "diffu_grpo":
        # Initialize and run trainer
        trainer = DiffuGRPOTrainer(
            args=grpo_config,
            model=model,
            peft_config=None if getattr(grpo_config, "resume_checkpoint_path", None) else peft_config,
            reward_funcs=reward_functions,
            train_dataset=train_set,
            optimizers=(optimizer, None),
            callbacks=[ResumeStepCallback(loaded_checkpoint_step, getattr(grpo_config, "max_steps", None))],
        )
    elif grpo_config.trainer == "spg":
        trainer = SPGTrainer(
            args=grpo_config,
            model=model,
            peft_config=None if getattr(grpo_config, "resume_checkpoint_path", None) else peft_config,
            reward_funcs=reward_functions,
            train_dataset=train_set,
            optimizers=(optimizer, None),
            callbacks=[ResumeStepCallback(loaded_checkpoint_step, getattr(grpo_config, "max_steps", None))],
        )
    else:
        raise ValueError(f"Invalid trainer: {grpo_config.trainer}")

    trainer.train()


if __name__ == "__main__":
    parser = TrlParser((DiffuGRPOConfig, ModelConfig))
    grpo_config, model_config = parser.parse_args_and_config()
    main(grpo_config=grpo_config, model_config=model_config)
