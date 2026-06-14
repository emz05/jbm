# Comparing Sanitization and Unlearning Defenses in LLMs
- evaluate filter and unlearning defenses against jailbreaking attacks on aligned LLMs
- attack vectors: direct prompting, GCG, PAIR
- unlearning defenses: GA, NPO, DPO, AP, GD
- filter defenses: perplexity, remove, smooth, synonym

## Setup
```bash
conda env create -f environment.yml
conda activate jbm
```
login with hugging face token

## Structure

unlearning-paper.pdf
- contains analysis over entire experiment

data/ 
- contains datasets loaded from hugging face in datasets.ipynb

unlearning_defense/
- contains losses, model, trainer for unlearning alg

filter_defense/
- contains five different filter defense alg

results/
- result: lists all jailbreaking prompts, responses, targets, etc
- summary: summarizes asr and rr

visualizations.ipynb
- provides graphics loaded from summary


