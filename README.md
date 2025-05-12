# Medical Condition Classifier 🧠🔍  
This project is a machine learning-based solution designed to predict medical conditions from patient data using NLP and deep learning. Built in Python and Jupyter Notebook, the model leverages BioBERT and an attention mechanism for high inference reliability.

## 📌 Project Description  
As part of my journey in applying AI to healthcare, I built this notebook to classify diseases based on natural language symptom descriptions. The goal is to provide a robust and interpretable model that not only achieves high accuracy but also consistently makes correct predictions during inference.

## 🧬 Key Features
- Uses **BioBERT (dmis-lab/biobert-base-cased-v1.1)** for contextual medical embeddings
- Includes an **attention-based classifier** for enhanced interpretability
- Supports **free-text symptom input** for real-world flexibility
- Post-processing using **symptom-disease overlap matrix** for refined inference
- Includes **Monte Carlo Dropout** for robust uncertainty estimation

## 📂 Project Structure
📁 Shipping-Management-Using-Blockchain-main/
├── MAIN MODEL (1).ipynb # Final disease classification notebook
├── notebook9f04b8289f.ipynb # Auxiliary or experimental notebook
├── models/ # (Optional) Model checkpoints
├── data/ # Processed and raw datasets
└── README.md # Project overview and instructions


## ⚙️ How to Run
1. Clone the repository:
```bash
git clone https://github.com/yourusername/disease-prediction-bioBERT.git
cd disease-prediction-bioBERT

Install required packages:
pip install -r requirements.txt

Run the notebook:

Open MAIN MODEL (1).ipynb in Jupyter or VSCode.

Start from the top and execute all cells.

📊 Model Overview
Base model: BioBERT (transformers from HuggingFace)

Classifier: Custom attention layer on top of pooled output

Loss function: CrossEntropyLoss

Optimization: AdamW, learning rate scheduler

Evaluation: Accuracy, Precision, F1-score, and inference test cases

📈 Future Work
Integrate lab result embeddings alongside symptom descriptions

Add support for multilingual symptom input

Deploy as a web-based diagnostic tool using Flask or Streamlit

Explore reinforcement learning for interactive diagnosis

🤝 Acknowledgments
Inspired by real-world clinical datasets and use cases

BioBERT: https://github.com/dmis-lab/biobert

HuggingFace Transformers for model integration
