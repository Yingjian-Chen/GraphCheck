# GraphCheck: Breaking Long-Term Text Barriers with Extracted Knowledge Graph-Powered Fact-Checking
[![Static Badge](https://img.shields.io/badge/2502.16514-Arxiv?style=flat&logo=arxiv&logoColor=%23B31B1B&logoSize=auto&label=Arxiv&labelColor=%23FFFFFF&color=%23B31B1B)](https://arxiv.org/abs/2502.16514)



GraphCheck is a fact-checking method that integrates knowledge graphs (KGs) to enhance LLM-based fact-checking, specifically for long-form text. By addressing the limitations of LLMs in capturing complex entity relationships, GraphCheck overcomes issues related to overlooked factual errors. The method leverages graph neural networks (GNNs) to integrate representations from both the generated claim and the source document KGs, enabling fine-grained fact-checking within a single model call. This significantly improves efficiency in the fact-checking process.

## Environment Setup

To set up the environment, simply run the following command to install all the required dependencies:

```bash
pip install -r requirements.txt
```

## Data Preprocessing

Before running, you need to preprocess the data by building the knowledge graph (KG) for the dataset. 
To build the graph for a specific dataset, use the following command:

```bash
python graph_build.py --data_name <dataset_name>
```

## Inference

To run the inference and reproduce the results presented in the paper, you can use the provided script `run_inference.sh`. This script automates the process and ensures that the inference is run in the correct environment with all necessary parameters.

Simply execute the script with the following command:

```bash
bash run_inference.sh
```

## Training

To train the model, you first need to modify the configuration parameters in `./src/config.py`. This file contains the hyperparameters and settings required for training the model, such as learning rate, batch size and number of epochs.

After that, you can start the training process by running the following command:

```bash
python train.py
```
