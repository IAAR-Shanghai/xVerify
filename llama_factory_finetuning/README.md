# xVerify LLaMA Factory Fine-tuning

This directory contains configurations and scripts for fine-tuning large language models using [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) within the xVerify project.

## Table of Contents

- [Overview](#overview)
- [Supported Models](#supported-models)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Data Preparation](#data-preparation)
- [Fine-tuning with QLoRA](#fine-tuning-with-qlora)
- [Model Merging](#model-merging)
- [Inference](#inference)

## Overview

The xVerify project leverages LLaMA Factory to fine-tune various language models using QLoRA (Quantized Low-Rank Adaptation). This approach enables efficient fine-tuning of large language models with limited computational resources while maintaining performance quality.

## Supported Models

Our xVerify project has successfully fine-tuned the following models:

- **LLaMA-3.1/3.2 series**: Fine-tuned 1B, 3B, and 8B variants with domain-specific data
- **Qwen2/2.5 series**: Optimized variants including 0.5B, 1.5B, 3B, and 7B for various tasks
- **Gemma-2 series**: Enhanced 2B and 9B models with improved reasoning capabilities
- **GLM models**: Adapted GLM-4-9B and ChatGLM3-6B for domain-specific applications
- **Phi-4**: Fine-tuned for specialized use cases in our verification pipeline

## Getting Started

### Installation

1. Clone and install LLaMA Factory:

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

2. Install recommended dependencies:
```bash
pip install -e ".[bitsandbytes]"  # Required for QLoRA training
pip install -e ".[vllm]"          # Recommended for high-performance inference
```

### Data Preparation

xVerify training datasets are organized and accessed through the [`dataset_info.json`](./data/dataset_info.json) configuration file:

1. Place your training datasets in the appropriate data directory:
   ```
   ./data/your_dataset_name.json
   ```

2. Register your datasets in the `data/dataset_info.json` file:
   ```json
    {
    "train_dataset_name": {
        "file_name": "mnt/path/to/train_dataset_name.json"
        }
    }
   ```

3. Reference your dataset in the training configuration:
   ```yaml
   dataset: train_dataset_name
   dataset_dir: ./data
   ```
The dataset loader will automatically locate and load your data based on these configurations.
## Fine-tuning with QLoRA

1. Customize the QLoRA configuration file at [`train_qlora_sft.yaml`](./scripts/train_qlora_sft.yaml). The parameters in the example script are the recommended settings we use for fine-tuning all xVerify models:
```yaml
model_name_or_path: ./models/base_model
dataset: your_dataset_name
template: gemma  # Select appropriate template for your model
output_dir: ./outputs/your-experiment-name
```

2. Launch the fine-tuning process:
```bash
llamafactory-cli train ./scripts/train_qlora_sft.yaml
```

## Model Merging

After fine-tuning, merge the LoRA adapter with the base model:

1. Configure the merge settings in [`merge_qlora_sft.yaml`](./scripts/merge_qlora_sft.yaml):
```yaml
model_name_or_path: ./models/base_model
adapter_name_or_path: ./outputs/your-experiment-name
export_dir: ./models/your-merged-model-name
```

2. Execute the merging process:
```bash
llamafactory-cli export ./scripts/merge_qlora_sft.yaml
```

This creates a fully-usable model that combines the base model's knowledge with your fine-tuned adaptations.

## Inference

To evaluate the performance, we recommend using [vLLM](https://docs.vllm.ai/en/latest/) for deployment and inference:

```bash
# Basic deployment
vllm serve --model ./models/your-merged-model --tensor-parallel-size 1
# High-throughput configuration
vllm serve --model ./models/your-merged-model --tensor-parallel-size 2 --max-model-len 8192
```