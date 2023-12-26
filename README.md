# Llamarizer
Fine-tuning and evaluation of Llama2-7bn model for news summary with a focus on factuality. Part of Advanced NLP 2023 @ ITU Copenhagen course work. Tested and developed on Python 3.10 and Ubuntu 22, NVidia A100 GPU.
See relevant links below:
- [Base model](https://huggingface.co/ernlavr/llama-2-7bn-xsum-lora-adapter)
- [Weights and Biases](https://wandb.ai/ernlavr/adv_nlp2023/sweeps)
- [Paper](https://ernlavr.github.io/portfolio/portfolio-5/)

## To run
0. Start by cloning `git clone <REPO> --recursive`, the `--recursive` flag is 
for cloning `BARTScore` evaluation repository
1. Start by creating the conda environment `conda env create -f environment.yml`
and update it as necessary
1. Run the following config files
    - `python3 main.py --args_path conf/train_llama_base.yaml` // Train the llamarizer baseline
    - `python3 main.py --args_path conf/train_DistilBERT_NLI.yaml` // train NLI module
    - `python3 main.py --args_path conf/train_llama_nli.yaml` // train the llamarizer-NLI (requires NLI module)
    - `python3 main.py --args_path conf/eval_llama_base.yaml` // evaluate the llamarizer baseline
    - `python3 main.py --args_path conf/eval_llama_nli.yaml` // evaluate the llamarizer-NLI

2. Additionally `run_hpc.sh` can be used for queueing the jobs on the HPC cluster although you must have your Conda environment setup and enabled upon queueing.

Additionally there are also configs for running param-sweeps prefixed `conf/sweep_*`. See those for more details and references.

If you wish to use HuggingFace and Weights&Biases integrations, you must have your API keys set up `WANDB_API_KEY` and `HF_TOKEN` in `main.py`.

## Dependencies
Conda environment should cover all the dependencies although the evaluation requires NLTK packages `stopwords` and `punkt` to be downloaded. This can be done by running the following in python:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```
