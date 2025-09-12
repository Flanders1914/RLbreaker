# RLbreaker Data Processing Scripts Documentation

This document describes two Python scripts used for generating and processing unaligned data in the RLbreaker project.

## Scripts Overview

### 1. generate_unaligned.py

**Purpose**: Generates unaligned responses using the Wizard-Vicuna-7B-Uncensored model for given prompts.

**Dependencies**:
- pandas
- torch
- transformers
- vicuna.jinja (chat template file)

**Input Files**:
- `./datasets/train.csv` - Training dataset with 'prompt' and 'target' columns
- `./datasets/val.csv` - Validation dataset with 'prompt' and 'target' columns
- `./datasets/test.csv` - Test dataset with 'prompt' and 'target' columns
- `vicuna.jinja` - Chat template for formatting conversations

**Output Files**:
- `./datasets/unalign_train.csv` - Generated responses for training data
- `./datasets/unalign_val.csv` - Generated responses for validation data
- `./datasets/unalign_test.csv` - Generated responses for test data

**Key Functions**:
- `extract_response(text)`: Extracts the assistant's response from the generated text
- `generate_response(messages)`: Uses the LLM to generate responses based on input messages
- `main(split)`: Main function that processes a data split and saves results

**Usage**:
```bash
# Set GPU device and run
CUDA_VISIBLE_DEVICES=0 python generate_unaligned.py
```

**Configuration**:
- Model: QuixiAI/Wizard-Vicuna-7B-Uncensored
- Max length: 512 tokens
- Precision: FP16
- Device mapping: Auto (handles multi-GPU/CPU)

### 2. fix_non_jailbreaking_data.py

**Purpose**: Post-processes the generated unaligned data by identifying and fixing non-jailbreaking responses that are too short.

**Dependencies**:
- pandas

**Input Files**:
- `./datasets/unalign_{data_split}.csv` - Generated unaligned data (output from generate_unaligned.py)
- `./datasets/advbench.csv` - Reference dataset for finding replacement responses

**Output Files**:
- `./datasets/processed_unalign_{data_split}.csv` - Processed data with fixed responses

**Key Logic**:
1. Identifies responses that are too short (less than 2x the question length)
2. Attempts to find replacement responses in the reference dataset (advbench.csv)
3. Matches questions case-insensitively and ensures replacement responses are substantial (>50 characters)

**Current Configuration**:
- Processes validation split (`data_split = "val"`)
- Threshold: Response length < 2 × question length
- Minimum replacement response length: 50 characters

**Usage**:
```bash
python fix_non_jailbreaking_data.py
```

**Statistics Reported**:
- Total number of rows processed
- Number of non-jailbreaking responses identified
- Number of responses successfully fixed from reference dataset


## Workflow

1. **Data Generation**: Run `generate_unaligned.py` to generate responses for all data splits
2. **Data Processing**: Run `fix_non_jailbreaking_data.py` to identify and fix short/inadequate responses
3. **Result**: Clean, processed datasets ready for training/evaluation

## Notes

- The generation script uses GPU acceleration (configure with CUDA_VISIBLE_DEVICES)
- The fixing script currently processes only the validation split (modify `data_split` variable for other splits)
- Chat template formatting follows the Vicuna conversation format
- The model generates responses by continuing from initial assistant prompts
