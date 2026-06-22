"""
SFT/lora_config.py

Provides a helper function to create a PEFT LoRA (Low-Rank Adaptation) configuration.

This module is responsible for converting the LoRA parameters from the configuration
into a `peft.LoraConfig` object that can be used to adapt a Hugging Face model
for efficient fine-tuning.
"""

from peft import LoraConfig, TaskType
from .config import LoraConfigData  # Import the dataclass


def get_lora_config(lora_cfg_data: LoraConfigData) -> LoraConfig:
    """
    Creates and returns a PEFT LoRA configuration for Causal Language Modeling.

    This function takes a LoraConfigData object and uses its parameters to
    instantiate a `peft.LoraConfig`. This configuration is essential for applying
    LoRA to the base model, enabling parameter-efficient fine-tuning.

    Args:
        lora_cfg_data (LoraConfigData): A dataclass object containing LoRA parameters:
            - r: The rank of the LoRA matrices.
            - lora_alpha: The alpha parameter for LoRA scaling.
            - lora_dropout: The dropout probability for LoRA layers.
            - target_modules: A list of module names to apply LoRA to.

    Returns:
        LoraConfig: A `peft.LoraConfig` object configured for `CAUSAL_LM` tasks.
    """
    return LoraConfig(
        r=lora_cfg_data.r,
        lora_alpha=lora_cfg_data.lora_alpha,
        lora_dropout=lora_cfg_data.lora_dropout,
        # Specify the task type as Causal Language Modeling, which is standard
        # for auto-regressive models like MedGemma.
        task_type=TaskType.CAUSAL_LM,
        target_modules=lora_cfg_data.target_modules,
        modules_to_save=lora_cfg_data.modules_to_save,
        # 'bias="none"' is a common practice to only train the LoRA weights (A and B matrices)
        # and not the bias terms, further improving efficiency.
        bias="none",
    )
