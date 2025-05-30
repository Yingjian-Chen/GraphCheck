# GraphCheck: Breaking Long-Term Text Barriers with Extracted Knowledge Graph-Powered Fact-Checking [ACL 2025]
[![ArXiv](https://img.shields.io/badge/2502.16514-Arxiv?style=flat&logo=arxiv&logoColor=%23B31B1B&logoSize=auto&label=Arxiv&labelColor=%23FFFFFF&color=%23B31B1B&link=https%3A%2F%2Farxiv.org%2Fabs%2F2502.16514
)](https://arxiv.org/pdf/2502.16514)

GraphCheck is a fact-checking method that integrates knowledge graphs (KGs) to enhance LLM-based fact-checking, specifically for long-form text. By addressing the limitations of LLMs in capturing complex entity relationships, GraphCheck overcomes issues related to overlooked factual errors. The method leverages graph neural networks (GNNs) to integrate representations from both the generated claim and the source document KGs, enabling fine-grained fact-checking within a single model call. This significantly improves efficiency in the fact-checking process.

<p align="center">
  <img src="Figs/graphcheck-framework.svg" style="max-width:100%; height:auto;">
</p>

## 📚 Datasets

We construct the training and evaluation datasets under the `/dataset` directory.

- **Training**:  
  Based on 14K synthetic samples generated from [**MiniCheck**](https://github.com/Liyan06/MiniCheck?tab=readme-ov-file#description), we extract knowledge graph triples from both documents and claims to construct the training set:
  - `MiniCheck_Train`

- **Evaluation**:  
  We use reconstructed evaluation benchmarks:
  - `AggreFact-XSum`
  - `AggreFact-CNN`
  - `summeval`
  - `ExpertQA`
  - `COVID-Fact`
  - `SCIFact`
  - `PubHealth`

Each data entry contains:
- `doc_text`: original document text  
- `claim_text`: the associated claim  
- `doc_kg`: extracted knowledge graphs from the document  
- `claim_kg`: extracted knowledge graphs from the claim  
- `label`: ground-truth label


## ⚙️ Environment Setup

To set up the environment, simply run the following command to install all the required dependencies:

```bash
pip install -r requirements.txt
```

## 🛠️ Data Preprocessing

Before running, you need to preprocess the data by building the knowledge graph (KG) for each dataset, use the following command:

```bash
bash graph_building.sh
```

## 🧠 Training

To start training, simply run:

```bash
bash train.sh
```
You can customize the training by modifying the configuration file at `./src/config.py`. This file includes key hyperparameters and settings such as learning rate, batch size, and number of epochs.

## 🚀 Quick Evaluation

To quickly evaluate the model, simply run:

   ```bash
   bash evaluate.sh
   ```

## 📌 Citation

If you find our project helpful, please feel free to leave a ⭐ and cite our paper:

```bibtex
@article{chen2025graphcheck,
  title={GraphCheck: Breaking Long-Term Text Barriers with Extracted Knowledge Graph-Powered Fact-Checking},
  author={Chen, Yingjian and Liu, Haoran and Liu, Yinhong and Yang, Rui and Yuan, Han and Fu, Yanran and Zhou, Pengyuan and Chen, Qingyu and Caverlee, James and Li, Irene},
  journal={arXiv preprint arXiv:2502.16514},
  year={2025}
}
