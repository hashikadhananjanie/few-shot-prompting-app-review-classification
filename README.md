# Few-Shot Prompt-Tuning of Language Models for App Review Classification: An Evaluation Study
This repository contains the code of the paper **Few-Shot Prompt-Tuning of Language Models for App Review Classification: An Evaluation Study**.

## Overview

This study focuses on evaluating few-shot prompt-tuning of Language Models in classifying developer-relevant information 
from app reviews. The experiments were conducted using different settings and one scripts for each Language 
Models (RoBERTa, T5 and GPT-2) used in this study is provided here, considering the three datasets used as well.

These scripts can be modified according to the requirements. For eg:

- modify training, test and validation dataset splitting strategies
- update prompt template design
- update verbalizer design

### Instructions to run the code

Clone the repository from GitHub.

```
git clone https://github.com/hashikadhananjanie/few-shot-prompting-app-review-classification.git
```

Install required libraries mentioned in `requirements.txt` file using pip install.
For eg:
```
pip install openprompt
```

Download publicly available labeled app review datasets from their respective sources and place them in the folder 
where you are running the Python scripts.

Run python scripts
```
python RQ1_MD1_DS1_VB1_PT1_EX4.py
```