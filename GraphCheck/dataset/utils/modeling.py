import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
import numpy as np

pretrained_repo = 'sentence-transformers/all-roberta-large-v1'
batch_size = 1024  # Adjust the batch size as needed


class Dataset(torch.utils.data.Dataset):
    def __init__(self, input_ids=None, attention_mask=None):
        super().__init__()
        self.data = {
            "input_ids": input_ids,
            "att_mask": attention_mask,
        }

    def __len__(self):
        return self.data["input_ids"].size(0)

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = index.item()
        batch_data = dict()
        for key in self.data.keys():
            if self.data[key] is not None:
                batch_data[key] = self.data[key][index]
        return batch_data


class Sentence_Transformer(nn.Module):

    def __init__(self, pretrained_repo):
        super(Sentence_Transformer, self).__init__()
        print(f"inherit model weights from {pretrained_repo}")
        self.bert_model = AutoModel.from_pretrained(pretrained_repo)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        data_type = token_embeddings.dtype
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(data_type)
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_ids, att_mask):
        bert_out = self.bert_model(input_ids=input_ids, attention_mask=att_mask)
        sentence_embeddings = self.mean_pooling(bert_out, att_mask)

        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings


def load_model():

    model = Sentence_Transformer(pretrained_repo)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_repo)

    # data parallel
    if torch.cuda.device_count() > 1:
        print(f'Using {torch.cuda.device_count()} GPUs')
        model = nn.DataParallel(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device


def load_text2embedding(model, tokenizer, device, text):
    if len(text) == 0:
        return torch.zeros((0, 1024))

    encoding = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    dataset = Dataset(input_ids=encoding.input_ids, attention_mask=encoding.attention_mask)

    # DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Placeholder for storing the embeddings
    all_embeddings = []

    # Iterate through batches
    with torch.no_grad():

        for batch in dataloader:
            # Move batch to the appropriate device
            batch = {key: value.to(device) for key, value in batch.items()}

            # Forward pass
            embeddings = model(input_ids=batch["input_ids"], att_mask=batch["att_mask"])

            # Append the embeddings to the list
            all_embeddings.append(embeddings)

    # Concatenate the embeddings from all batches
    all_embeddings = torch.cat(all_embeddings, dim=0).cpu()

    return all_embeddings