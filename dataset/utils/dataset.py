import pandas as pd
import torch
from torch.utils.data import Dataset
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PATH = f'{project_root}'

def get_dataset(dataset_name):
    result_df = pd.read_pickle(f'{PATH}/extracted_KG/{dataset_name}/{dataset_name}.pkl')
    docs = result_df['doc_text'].values
    claims = result_df['claim_text'].values
    doc_kgs = result_df['doc_kg'].values
    claim_kgs = result_df['claim_kg'].values
    labels = result_df['label'].values
    return docs, claims, doc_kgs, claim_kgs, labels

class KGDataset(Dataset):
    def __init__(self, dataset_name):
        super().__init__()
        self.dataset_name = dataset_name
        self.docs, self.claims, self.doc_kgs, self.claim_kgs, self.labels = get_dataset(self.dataset_name)
        self.prompt = "Question: Does the Document support the Claim? Please Answer in one word in the form of \'support\' or \'unsupport\'.\n\n"


    def __len__(self):
        """Return the len of the dataset."""
        return len(self.docs)

    def __getitem__(self, index):

        doc, claim, doc_kgs_text, claim_kgs_text, label = self.docs[index], self.claims[index], self.doc_kgs[index], self.claim_kgs[index],self.labels[index]
        text = f'{self.prompt}\nClaim: {claim}\nDocument: {doc}'
        claim_kg = torch.load(f'{PATH}/extracted_KG/{self.dataset_name}/graphs/claim/{index}.pt')
        doc_kg = torch.load(f'{PATH}/extracted_KG/{self.dataset_name}/graphs/doc/{index}.pt')
        if label == 1:
            label = 'support'
        else:
            label = 'unsupport'

        return {
            'id': index,
            'label': label,
            'claim_kg': claim_kg,
            'doc_kg': doc_kg,
            'claim_kg_text': claim_kgs_text,
            'doc_kg_text': doc_kgs_text,
            'text': text,
            'index': index,
            'dataset': self.dataset_name
        }

    def get_idx_split(self):

        # Load the saved indices
        with open(f'{PATH}/extracted_KG/{self.dataset_name}/split/train_indices.txt', 'r') as file:
            train_indices = [int(line.strip()) for line in file]

        with open(f'{PATH}/extracted_KG/{self.dataset_name}/split/val_indices.txt', 'r') as file:
            val_indices = [int(line.strip()) for line in file]

        with open(f'{PATH}/extracted_KG/{self.dataset_name}/split/test_indices.txt', 'r') as file:
            test_indices = [int(line.strip()) for line in file]

        return {'train': train_indices, 'val': val_indices, 'test': test_indices}

if __name__ == '__main__':
    dataset = KGDataset()
    data = dataset[0]
    for k, v in data.items():
        print(f'{k}: {v}')
