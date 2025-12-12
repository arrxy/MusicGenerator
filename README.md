# CSGY 6923 Project: Music Generation using Transformer-based

# Models

## Overview

This repository contains the code and configuration for the CSGY 6923 project investigating scaling
laws in symbolic music generation using decoder-only Transformer models (nanoGPT architecture
adapted for ABC notation).

The project includes custom ETL pipelines for MIDI to ABC conversion, a family of scaled Transformer
models, and scripts for training, evaluation, and music generation.

## 1. Environment Setup

Critical Requirement: Training and full generation must be performed on high-performance machines
with sufficient GPU memory (minimum 40GB VRAM recommended for XL models).

Tested Hardware: NVIDIA A100 / H100 / H200 GPU instances (specifically Digital Ocean GPU Cluster).

**1.1 Dependencies and Installation**

1. Clone the Repository:

```
git clone https://github.com/arrxy/MusicGenerator.git
cd MusicGenerator
```
1.1. Setup Git LFS:

1.1.1. Homebrew
```aiignore
brew install git-lfs
```
1.1.2 Debian
```aiignore
sudo apt-get install git-lfs
```
1.2
```aiignore
git lfs install
git lfs pull
```
3. Install midi2abc: This is required for the data conversion pipeline. Install the command-line tool
    on your machine. _(Installation process varies by OS, e.g.,_ sudo apt-get install midi2abc _on_
    _Debian/Ubuntu)._
3. Install Python Dependencies: This project uses uv for dependency management.


```
# Install uv if you don't have it
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
# Sync dependencies from pyproject.toml
uv sync
```
## 2. Data Preparation Pipeline

**2.1 Download and Extraction**


1. Download the Lakh MIDI Dataset (LMD): The project relies on the full LMD dataset. Download
    and extract it:

```
# Download the dataset (using the provided link)
wget [https://colinraffel.com/projects/lmd/Lakh_MIDI_Dataset.zip](https://colinraff
unzip Lakh_MIDI_Dataset.zip
```
2. Create Directory Structure:

```
mkdir -p data/raw
mkdir -p data/processed
```
3. Place Data: Extract the LMD files into the data/raw directory.

**2.2 Tokenization and Preprocessing**

The custom Python script handles MIDI conversion, cleaning, filtering, and tokenization, producing the
final vocab.json and tokenized .bin files used for training.

Estimated Time: This process takes approximately 2-5 minutes on the target machine.

```
uv run python main.py
```
_(The_ main.py _script executes the entire Extract-Transform-Load pipeline as described in the report,
including tokenization and train/val/test split generation.)_

## 3. Training and Evaluation

**3.1 Training the Best Model (XL)**

The primary training script is train_transformer.py. This script is configured to train the XL model
(n_layer=12, n_embd=768) based on the scaling study results.

```
uv run python train_transformer.py
```
_(Ensure the script configuration matches the "XL" settings and targets the optimal token budget (
Billion tokens) documented in the project report.)_

**3.2 Generating Music Samples**


After training and saving the final checkpoint (ckpt_XL_extended.pt), use the following scripts for
music generation:

**A. Unconditional Generation (From Scratch)**

This generates music starting only from the <start> token.

```
uv run python generate.py
```
**B. Conditional Generation (Continuation/Prompting)**

This script requires defining an initial sequence (ABC prefix) _inside the script_
(generate_continuation.py) to guide the model's output.

```
# First, open the script and edit the initial sequence prompt:
# nano generate_continuation.py
# Then run the script:
uv run python generate_continuation.py
```
Note: For generation, refer to the configuration variables (MODEL_SIZE, CHECKPOINT_PATH, MAX_NEW_TOKENS, TEMPERATURE, etc.) located within the respective Python scripts (generate.py or generate_continuation.py).
