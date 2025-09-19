# RLbreaker Codebase Modifications

## Major Modifications

### 1. Enhanced Unaligned Data

The unaligned responses in advbench.csv which is used by the original codebase contains around a quarter of refuse-to-answer responses.

To address this quality issue, I implemented a new Unaligned Data Generation System that produces higher-quality unaligned responses for training. The unaligned datasets have already been generated and are ready to use for both training and testing purposes. The following documentation describes the system's implementation details for reference.

#### 1.1 Unaligned Data Generation System
**New Files Added:**
- `generate_unaligned.py` - Automated generation of unaligned responses
- `fix_non_jailbreaking_data.py` - Post-processing to fix inadequate responses
- `prepare_datasets.py` - Dataset preparation and management utilities
- `analyze_dataset.py` - Dataset analysis and inspection tools

**Key Features:**
- **Automated Response Generation**: Uses Wizard-Vicuna-7B-Uncensored model to generate realistic unaligned responses for training data
- **Quality Control**: Automatically identifies and fixes responses that are too short or non-jailbreaking
- **Reference Dataset Integration**: Leverages advbench.csv as a reference to improve response quality
- **Chat Template Support**: Includes `vicuna.jinja` template for proper conversation formatting

**Technical Details:**
- Model: QuixiAI/Wizard-Vicuna-7B-Uncensored
- Precision: FP16 for memory efficiency
- Max tokens: 512
- Quality threshold: Response length > 2× question length
- Minimum replacement response: 50 characters

#### 1.2 Expanded Dataset Collection
Our data splits and the corresponding unaligned respones are presented in the following files.

**Dataset Files:**
- `processed_unalign_train.csv`, `processed_unalign_val.csv`, `processed_unalign_test.csv` - Clean, processed datasets
- `unalign_train.csv`, `unalign_val.csv`, `unalign_test.csv` - Raw generated datasets
- `train.csv`, `val.csv`, `test.csv` - Split datasets for systematic training/evaluation

### 2. Extended Model Support

#### 2.1 Expanded Model Compatibility
**Modified Files:**
- `llm_utils/creat_model.py` - Enhanced model creation with broader API support

**Supported Models:**
- **OpenAI Models**: GPT-3.5-turbo variants, GPT-4o series
- **Meta Models**: Llama-3.2-11B-Vision-Instruct, Meta-Llama-3-8B-Instruct, Llama-2-70b-chat-hf
- **Mistral Models**: Mixtral-8x7B-Instruct-v0.1, Mixtral-8x22B-Instruct-v0.1
- **Qwen Models**: Qwen/Qwen3-14B
- **Google Models**: google/gemini-2.5-flash
- **Other Models**: openai/gpt-oss-20b

#### 2.2 API Modification
The original codebase would use reasoning model for qwen and gemini, I have turned it off

- **Reasoning off**: reasoning_effort for Gemini/Qwen models are disabled

### 3. Evaluation and Analysis Modification
Use GPT-4o as judge to align with other methods

1. **GPT-4o Judge**: Use GPT-4o as the GPT-judge
4. **Evaluation Logging**: The evaluation result is logged and saves in {model_name}.json

### 4. Workflow Modification
The original codebase only set a max_query = 10000 for the entire testing run, which would be unfair for comparison. 
I have modified the code to support a limit number of trial attempts for each question.

Controlled by max_attempts_per_question in the test_policy.py

## Usage Guide

### 1. Environment Setup

The following environment setup may not work in another server.
I am Sorry for your inconvenience.

```bash
# Create conda environment
conda create --name RLbreaker python=3.8
conda activate RLbreaker

# Install dependencies
pip install -r requirements.txt
```

### 2. Training

```bash
CUDA_VISIBLE_DEVICES=6 python train_policy.py \
--index=0 \
--env-name=Qwen3-14B \
--target_model=Qwen/Qwen3-14B \
--model_path=gpt-3.5-turbo \
--openai_key=[YOUR_OPENAI_KEY] \
--deepinfra_key=[YOUR_DEEPINFRA_KEY]
```

### 3. Testing and Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python test_policy.py \
--index=0 \
--target_model=Qwen/Qwen3-14B \
--model_path=gpt-3.5-turbo \
--ckpt_path=trained_models/ppo/Qwen3-14B_final.pt \
--max_query=10000 \
--max_attempts_per_question=50 \
--openai_key=[YOUR_OPENAI_KEY] \
--deepinfra_key=[YOUR_DEEPINFRA_KEY]
```

### 4. Results Analysis

```bash
python analyze_results.py \
--file_path=datasets/eval/RL_Qwen3-14B_0_responses_none.csv \
--target_model=Qwen/Qwen3-14B \
--openai_key=[YOUR_OPENAI_KEY] \
--deepinfra_key=[YOUR_DEEPINFRA_KEY] \
--cuda_id=0
```
