

# EQUATOR Evaluator

## Overview

The **EQUATOR Evaluator** is a robust framework designed to systematically evaluate the factual accuracy and reasoning capabilities of large language models (LLMs). Unlike traditional evaluation methods, which often prioritize fluency over accuracy, this tool employs a **deterministic scoring system** that ensures precise and unbiased assessment of LLM-generated responses.

This repository implements the methodology described in the research paper "EQUATOR: A Deterministic Framework for
Evaluating LLM Reasoning with Open-EndedQuestions. # v1.0.0-beta"(Bernard et al., 2024). By leveraging vector databases and smaller, locally hosted LLMs, the LLM Evaluator bridges the gap between scalability and accuracy in automated assessments.

---

## Key Features

1. **Deterministic Scoring**: Assigns binary scores (100% or 0%) based solely on factual correctness.
2. **Vector Database Integration**: Embeds open-ended questions and human-evaluated answers for semantic matching.
3. **Automated Evaluation**: Uses smaller LLMs to provide scalable and efficient assessments.
4. **Bias Mitigation**: Eliminates scoring biases related to linguistic fluency or persuasion.
5. **Cost Efficiency**: Optimizes token usage, significantly reducing operational costs for evaluation.

---

## Why LLM Evaluator?

Traditional methods, like multiple-choice or human evaluation, fail to capture the nuanced reasoning and factual accuracy required in high-stakes domains such as medicine or law. The LLM Evaluator:

- Focuses on **factual correctness** over linguistic style.
- Reduces reliance on human evaluators by automating the grading process.
- Provides insights into where LLMs fall short, enabling targeted improvements in model training.

---

## Methodology

### 1. Deterministic Scoring Framework
The scoring framework evaluates LLM-generated answers against a vector database of human-evaluated responses. It follows these steps:
1. **Embed Inputs**: Convert questions and answers into vector embeddings using models like `all-minilm`.
2. **Retrieve Closest Match**: Identify the most semantically similar answer key using cosine similarity.
3. **Binary Scoring**: Assign 100% if the student’s answer matches the answer key; otherwise, 0%.

### 2. Vector Database
The vector database, implemented with ChromaDB, stores embeddings of open-ended questions and their corresponding answer keys. This database serves as the single source of truth for evaluations.

### 3. Evaluator LLM
A smaller LLM (e.g., LLaMA 3.2B) acts as the evaluator, ensuring strict adherence to the scoring criteria while reducing computational overhead.

---

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/raymondbernard/equator.git
   cd equator 
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment**
   - Create a `.env` file with necessary API keys and configurations.
   - Example:
     ```
     OPENROUTER_KEY=your_key_here
     ```

---

## User Instructions: Variables

This section allows users to configure various settings of the LLM Evaluator:

1. **Static Analysis Only**:
   - If you want to run only the static analysis, comment out the `llm_evaluate` in the execution list within `main.ipynb`.

2. **Model Selection**:
   - You can specify the models you wish to evaluate. The current setup uses OpenAI's API through OpenRouter, which offers access to approximately 275 models.
   - Free models are limited to about 200 calls per day. For extended usage, consider creating a paid OpenRouter account to avoid this limitation. We have tested the free models to ensure the code runs as expected.

3. **`keepVectorDB` Setting**:
   - Set this to `true` if you have already input the data and want to avoid re-importing it. 
   - The evaluator uses `linguistic_benchmark.json` as the source of truth for grading. You can customize this file to align with your specific use case.

4. **Directory Structure**:
   - To maintain a consistent directory structure, you can hard-code the date in the configuration.

---

## Setting Up a Virtual Environment

### On Windows

1. **Create a Virtual Environment**
   ```bash
   python -m venv .venv
   ```

2. **Activate the Virtual Environment**
   ```bash
   .venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Deactivate the Environment**
   ```bash
   deactivate
   ```

### On Linux/MacOS

1. **Create a Virtual Environment**
   ```bash
   python3 -m venv .venv
   ```

2. **Activate the Virtual Environment**
   ```bash
   source .venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Deactivate the Environment**
   ```bash
   deactivate
   ```

---

## Usage

### Running the Program

1. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```
2. **Open the Notebook**
   Navigate to `main.ipynb` in your browser.

3. **Execute the Cells**
   Follow the notebook's sequence to:
   - Load the dataset.
   - Embed questions and answers.
   - Run the evaluator.
   - Generate results and visualizations.

### Viewing Results

Results, including scores and explanations, are saved in the specified output directory as JSON files. Each entry includes:
- Question
- Model-generated answer
- Score
- Explanation for the score

---

## Example Dataset

The included dataset (`linguistic_benchmark.json`) features open-ended questions across various categories, including puzzles, spatial reasoning, and logic. This diversity tests the reasoning capabilities of LLMs comprehensively.
Here’s the updated section for the `README.md` to reflect the two datasets and instructions on renaming the larger one when ready to use:

---

## Example Dataset

The repository includes two datasets to test the reasoning capabilities of LLMs:

1. **Default Dataset**: 
   - The file `linguistic_benchmark.json` contains open-ended questions across various categories, such as puzzles, spatial reasoning, and logic. This smaller dataset is ideal for quick tests or debugging.

2. **Extended Dataset**:
   - A larger dataset with 1,005 QA pairs is provided as `linguistic_benchmark.json.bak`. This file contains more comprehensive benchmarks for a detailed evaluation.

### Using the Larger Dataset

To use the larger dataset, simply rename the `.bak` file:
```bash
mv linguistic_benchmark.json.bak linguistic_benchmark.json
```

Once renamed, the evaluator will use the larger dataset to run all benchmarks. Ensure sufficient computational resources and model API quotas before using the extended dataset.


## Contributions

### Authors
- Raymond Bernard (Independent Researcher)
- Shaina Raza, Ph.D. (Vector Institute)
- Subhabrata Das, PhD (JP Morgan Chase)
- Raul Murugan (Columbia University)

---

## Future Work

- Expand the vector database to include more diverse datasets.
- Optimize the embedding and retrieval process for larger-scale deployments.
- Investigate additional scoring criteria for complex reasoning tasks.

---
- *Acknowledgment*: We extend our gratitude to James Huckle for inspiring our work.  
- We have incorporated elements from [https://github.com/autogenai/easy-problems-that-llms-get-wrong](https://github.com/autogenai/easy-problems-that-llms-get-wrong).  
- Our approach advances the field by simplifying the benchmarking process through our capability to score open-ended questions effectively.  
- Rather than benchmarking multiple models across disparate APIs, we leverage OpenRouter.ai's unified API, using the OpenAI SDK, which provides access to over 270 models for comprehensive benchmarking.  ## Citation
If you use this framework in your research, please cite:

```
@article{bernard2024EQUATOR,
  title={EQUATOR: A Deterministic Framework for Evaluating LLM Reasoning with Open-Ended Questions. # v1.0.0-beta},
  author={Bernard, Raymond and Raza, Shaina  Das, Subhabrata and  Murugan, Raul},
  journal={Preprint},
  year={2024}
}
```

