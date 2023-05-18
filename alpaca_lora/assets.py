from dagster import asset, Config
from .utils.finetune import FinetuneConfig, finetune_alpaca_lora_model
from .utils.checkpoints import ExportCheckpointConfig, export_alpaca_lora_checkpoint
from .resources import DataDirectory
from pathlib import Path
import requests
from subprocess import run

# TODO: quantize
# TODO: convert to ggml
# TODO: download llama hf
# TODO: convert from original weights


class FoundationModelWeightsConfig(Config):
    huggingface_repo: str = "decapoda-research/llama-7b-hf"


@asset
def foundation_model_weights(config: FoundationModelWeightsConfig, data_dir: DataDirectory) -> Path:
    try:
        run(['git', 'lfs', 'install'], check=True)
    except:
        raise RuntimeError("git lfs is not installed so we cannot clone the model weights")
    output_dir = data_dir.subdir("foundation_model_weights")
    run(
        ['git', 'clone', '--depth', '1', f"https://huggingface.co/{config.huggingface_repo}", "."],
        cwd=output_dir,
        check=True,
    )
    return output_dir


class InstructionDataConfig(Config):
    url: str = (
        'https://huggingface.co/datasets/yahma/alpaca-cleaned/resolve/main/alpaca_data_cleaned.json'
    )


@asset
def instruction_data(config: InstructionDataConfig, data_dir: DataDirectory) -> Path:
    output_file = data_dir.subdir("instruction_data") / "instruction_data.json"
    with open(output_file, "w") as f:
        # f.write(requests.get(config.url, allow_redirects=True).content)
        import json

        json.dump(
            [
                {
                    "instruction": "Give three tips for staying healthy.",
                    "input": "",
                    "output": "1. Eat a balanced and nutritious diet: Make sure your meals are inclusive of a variety of fruits and vegetables, lean protein, whole grains, and healthy fats. This helps to provide your body with the essential nutrients to function at its best and can help prevent chronic diseases.\n\n2. Engage in regular physical activity: Exercise is crucial for maintaining strong bones, muscles, and cardiovascular health. Aim for at least 150 minutes of moderate aerobic exercise or 75 minutes of vigorous exercise each week.\n\n3. Get enough sleep: Getting enough quality sleep is crucial for physical and mental well-being. It helps to regulate mood, improve cognitive function, and supports healthy growth and immune function. Aim for 7-9 hours of sleep each night.",
                }
            ],
            f,
        )
    return output_file


@asset
def lora_weights(
    data_dir: DataDirectory, instruction_data: Path, foundation_model_weights: Path
) -> Path:
    output_dir = data_dir.subdir("lora_weights")
    finetune_alpaca_lora_model(
        FinetuneConfig(
            base_model=foundation_model_weights,
            output_dir=output_dir,
            data_path=instruction_data,
            val_set_size=0,
        )
    )
    return output_dir


@asset
def model_checkpoint(
    lora_weights: Path, data_dir: DataDirectory, foundation_model_weights: Path
) -> Path:
    output_dir = data_dir.subdir("model_checkpoint")
    export_alpaca_lora_checkpoint(
        ExportCheckpointConfig(
            base_model=foundation_model_weights,
            lora_weights=lora_weights,
            output_dir=output_dir,
        )
    )
    return output_dir
