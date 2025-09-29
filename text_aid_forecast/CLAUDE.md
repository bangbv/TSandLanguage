# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Context Aid Forecasting** research project that leverages Large Language Models (LLMs) for time series forecasting. The project explores how contextual information (captions/descriptions) affects LLM-based forecasting performance using both GPT-4 and LLaMA models.

## Environment Setup

### Dependencies
The project uses Conda for dependency management. Install the environment with:
```bash
conda env create -f environment.yml
conda activate TSandLangContext
```

### Required Environment Variables
```bash
export OPENAI_API_KEY=<your_api_key>
export OPENAI_API_BASE=<your_base_url>
export PYTHONPATH="${PYTHONPATH}:<project_root>/text_aid_forecast/"
```

## Core Commands

### Main Forecasting Scripts
All scripts are located in `./script/` directory:

1. **LLM-TIME without context**: `./script/llmtime_wo_context.sh`
2. **LLM-TIME with all context**: `./script/llmtime_wi_all_context.sh`
3. **LLM-TIME with specific captions**: `./script/llmtime_wi_captions.sh`
4. **Original scale LLM forecasting**: `./script/llm_ori_scale.sh`

### Direct Python Execution
```bash
# GPT-4 without context
python3 ./tsllm/main.py experiment=llmtime_wo_context model=gpt-4

# GPT-4 with all context
python3 ./tsllm/main_2.py experiment=llmtime_wi_all model=gpt-4

# LLaMA models
python3 ./tsllm/main_llama.py experiment=llmtime_wi_all_llama_2 model=llama
```

## Architecture Overview

### Core Modules

- **`tsllm/main.py`**: Primary entry point using Hydra configuration management
- **`tsllm/main_llama.py`**: LLaMA-specific forecasting pipeline with enhanced logging
- **`tsllm/main_2.py`**: Alternative pipeline for context-enhanced forecasting

### Data Processing Pipeline
1. **Dataset Loading** (`tsllm/datasets.py`, `tsllm/datasets_llama.py`): Loads time series datasets with descriptions
2. **Preprocessing** (`tsllm/pre_processing.py`, `tsllm/pre_processing_llama.py`):
   - Handles rescaling and normalization
   - Tokenization for LLM input
   - Series decomposition (trend, seasonal, residual components)
   - Fourier transform features
3. **Model Inference** (`tsllm/models/`): GPT-4 and LLaMA model implementations
4. **Serialization** (`tsllm/serialize.py`): Converts time series data to/from text format

### Configuration System
- **Hydra-based configuration** in `tsllm/config/`
- **Base config**: `tsllm/config/config.yaml`
- **Experiment configs**: `tsllm/config/experiment/` (different context scenarios)
- **Model configs**: `tsllm/config/model/` (GPT-4, LLaMA parameters)

### Key Features
- **Debug mode**: Set `debug_mode: True` in config for detailed logging
- **Test mode**: Set `is_test_mode: True` to use shortened time series (length=10)
- **Fourier analysis**: Set `is_fourier: True` for frequency domain features
- **Reasoning mode**: Set `is_reasoning: True` for enhanced LLM reasoning

### Output Structure
Results are saved to `./outputs/<date>/` as pickle files containing:
- Predicted samples and medians
- Model completions
- Input strings and metadata
- Scaling information

### Model Integration
- **GPT models**: Use OpenAI API with configurable temperature and test length
- **LLaMA models**: Local inference with batch processing support
- **Token utilities**: Smart batching and completion checking in `tsllm/token_utils.py`

### Data Flow
1. Load datasets with contextual descriptions
2. Decompose time series into trend/seasonal/residual components
3. Apply Fourier transforms if enabled
4. Generate text representations with optional context
5. Query LLM for forecasting
6. Deserialize predictions and apply inverse scaling
7. Save results with evaluation metrics

## Development Notes

### Testing
- Use `is_test_mode: True` for rapid development with shortened series
- Check `outputs/` directory for visualization examples in Jupyter notebooks

### Configuration Tips
- Experiment configs control context inclusion strategies
- Model configs set temperature, test length, and sampling parameters
- Use `num_of_samples` to control prediction ensemble size (20 for GPT, 96 for local models)

### Common Issues
- Ensure PYTHONPATH includes project root for cross-module imports
- Check OpenAI API key/base URL for GPT models
- Monitor token limits when using long context descriptions