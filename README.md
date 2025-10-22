# IMDb Review Sentiment Analysis (CS4824 Capstone)

**NOTE**: The notebook may not show any preview, if this is the case, please clone and open locally or through Google Colab.

## Overview
- Investigates long-form movie review sentiment classification using the IMDb dataset included in this repo (`Train.csv`, `Valid.csv`, `Test.csv`).
- Explores pre-trained transformer models (BERT-base-uncased, BERT-large-uncased, and RoBERTa-base) as baselines, then fine-tunes the strongest performer.
- Documents experiments, metrics, and visual analyses inside `Movie_Reviews.ipynb`.

## Repository Layout
- `Movie_Reviews.ipynb` – main notebook with exploratory data analysis (EDA), model training, evaluation, and parameter exploration and tuning.
- `Train.csv`, `Valid.csv`, `Test.csv` – labeled IMDb review splits used throughout the notebook.
- `requirements.txt` – Python dependencies for local execution.

## Getting Started
1. **Clone & enter the project**
	```bash
	git clone https://github.com/rkh4m/CS4824_Capstone_Project.git
	cd CS4824_Capstone_Project
	```
2. **Create/activate a virtual environment** (optional but recommended)
	```bash
	python -m venv capstone
	source capstone/bin/activate  # Windows Git Bash
	```
3. **Install dependencies**
	```bash
	pip install -r requirements.txt
	```

> **GPU note:** Fine-tuning transformer models benefits greatly from GPU acceleration. When running locally without a GPU, expect longer training times or consider using Google Colab (a one-click badge is embedded at the top of the notebook).

## Notebook Guide
- **Section 1 – EDA & Data Setup:** Inspects label balance, text length distributions, and missing values. Establishes context for tokenization choices.
- **Section 2 – BERT Family Evaluation:** Tokenizes reviews to 256 tokens and benchmarks three pre-trained transformers on test/validation splits without fine-tuning.
- **Section 3 – Fine-Tuning:** Applies consistent training arguments drawn from the original BERT paper, evaluates per model, and visualizes accuracy/F1 improvements plus loss curves.
- **Section 4 – Learning Rate Tuning:** Runs additional tests on BERT-base with warmup + linear decay to study how learning rate impacts F1.
- **Other Sections WIP**
- **Appendix:** Credits external references (HuggingFace docs, community guides) used to adapt training utilities.

## Running the Analysis
- **Local execution:** Launch Jupyter Lab/Notebook after installing requirements, open `Movie_Reviews.ipynb`, and run cells sequentially. Ensure the CSV files remain in the project root so relative paths resolve.
- **Google Colab:** Click the “Open in Colab” badge in the first cell. Upload or mount the dataset splits when prompted; the notebook contains hooks for Google Drive storage of model checkpoints and metrics.

## Outputs & Results
- Fine-tuned models achieve materially higher F1 scores versus their pre-trained baselines, confirming the value of task-specific adaptation. Detailed metrics tables and plots are rendered in Sections 2–4.
- Training logs, comparison charts (accuracy, F1, loss), and learning rate summaries are produced inside the notebook for direct inspection.

## Reproducibility Checklist
- Random seeds are controlled through the `transformers.TrainingArguments` defaults; reruns should produce consistent ordering but may vary slightly.
- Save directories and checkpoints assume a `/content/drive/...` layout when executed in Colab. Adjust paths if running elsewhere.

## Citation & Credits
- Dataset: [IMDb Movie Reviews for Sentiment Analysis](https://www.kaggle.com/datasets/columbine/imdb-dataset-sentiment-analysis-in-csv-format/data) (binary labels 0 = negative, 1 = positive).
- Model implementations and helper utilities reference HuggingFace Transformers.
