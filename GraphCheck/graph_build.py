import os
import sys
import torch
import pandas as pd
from dataset.utils.modeling import load_model, load_text2embedding
from tqdm import tqdm
from torch_geometric.data.data import Data
import os
import numpy as np
from sklearn.model_selection import train_test_split
import argparse

pythonpath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
print(pythonpath)
sys.path.insert(0, pythonpath)

def parse_args():
    parser = argparse.ArgumentParser(description="GraphCheck")

    parser.add_argument("--data_name", type=str, required=True, help="Name of the dataset folder")
    parser.add_argument("--project_root", type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), help="Root directory of the project")

    args = parser.parse_args()
    return args


def textualize_graph(graph):
    if not graph:
        nodes = pd.DataFrame(columns=['node_attr', 'node_id'])
        edges = pd.DataFrame(columns=['src', 'edge_attr', 'dst'])
        return nodes, edges

    nodes = {}
    edges = []

    for tri in graph:
        src, edge_attr, dst = tri

        # Replace null or empty values
        src = src.lower().strip() if src else " "
        edge_attr = edge_attr.lower().strip() if edge_attr else " "
        dst = dst.lower().strip() if dst else " "

        if src not in nodes:
            nodes[src] = len(nodes)
        if dst not in nodes:
            nodes[dst] = len(nodes)

        # Add edge to the edges list
        edges.append({
            'src': nodes[src],
            'edge_attr': edge_attr,
            'dst': nodes[dst],
        })

    # Convert nodes and edges to DataFrame
    nodes = pd.DataFrame(nodes.items(), columns=['node_attr', 'node_id'])
    edges = pd.DataFrame(edges)

    return nodes, edges


def step_one():
    # Create directories for storing claim and doc graphs
    os.makedirs(f'{path}/{data_name}/nodes/claim', exist_ok=True)
    os.makedirs(f'{path}/{data_name}/nodes/doc', exist_ok=True)
    os.makedirs(f'{path}/{data_name}/edges/claim', exist_ok=True)
    os.makedirs(f'{path}/{data_name}/edges/doc', exist_ok=True)

    for i, (claim_kg, doc_kg) in enumerate(tqdm(zip(claim_kgs, doc_kgs), total=len(dataset))):
        # Process claim graph
        claim_nodes, claim_edges = textualize_graph(claim_kg)
        claim_nodes.to_csv(f'{path}/{data_name}/nodes/claim/{i}.csv', index=False, columns=['node_id', 'node_attr'])
        claim_edges.to_csv(f'{path}/{data_name}/edges/claim/{i}.csv', index=False, columns=['src', 'edge_attr', 'dst'])

        # Process doc graph
        doc_nodes, doc_edges = textualize_graph(doc_kg)
        doc_nodes.to_csv(f'{path}/{data_name}/nodes/doc/{i}.csv', index=False, columns=['node_id', 'node_attr'])
        doc_edges.to_csv(f'{path}/{data_name}/edges/doc/{i}.csv', index=False, columns=['src', 'edge_attr', 'dst'])


def step_two():

    def _encode_graph():
        print('Encoding graphs...')
        # encoding claim
        os.makedirs(f'{path}/{data_name}/graphs/claim', exist_ok=True)
        for i in tqdm(range(len(dataset))):
            nodes = pd.read_csv(f'{path}/{data_name}/nodes/claim/{i}.csv')
            edges = pd.read_csv(f'{path}/{data_name}/edges/claim/{i}.csv')
            x = text2embedding(model, tokenizer, device, nodes.node_attr.tolist())
            e = text2embedding(model, tokenizer, device, edges.edge_attr.tolist())
            edge_index = torch.LongTensor([edges.src, edges.dst])
            data = Data(x=x, edge_index=edge_index, edge_attr=e, num_nodes=len(nodes))
            torch.save(data, f'{path}/{data_name}/graphs/claim/{i}.pt')

        # encoding doc
        os.makedirs(f'{path}/{data_name}/graphs/doc', exist_ok=True)
        for i in tqdm(range(len(dataset))):
            nodes = pd.read_csv(f'{path}/{data_name}/nodes/doc/{i}.csv')
            edges = pd.read_csv(f'{path}/{data_name}/edges/doc/{i}.csv')
            x = text2embedding(model, tokenizer, device, nodes.node_attr.tolist())
            e = text2embedding(model, tokenizer, device, edges.edge_attr.tolist())
            edge_index = torch.LongTensor([edges.src, edges.dst])
            data = Data(x=x, edge_index=edge_index, edge_attr=e, num_nodes=len(nodes))
            torch.save(data, f'{path}/{data_name}/graphs/doc/{i}.pt')

    model, tokenizer, device = load_model()
    text2embedding = load_text2embedding

    _encode_graph()

def generate_split(num_nodes, path):

    # Split the dataset into train, val, and test sets
    indices = np.arange(num_nodes)
    train_indices, temp_data = train_test_split(indices, test_size=0.4, random_state=42)
    val_indices, test_indices = train_test_split(temp_data, test_size=0.5, random_state=42)
    print("# train samples: ", len(train_indices))
    print("# val samples: ", len(val_indices))
    print("# test samples: ", len(test_indices))

    # Create a folder for the split
    os.makedirs(path, exist_ok=True)

    # Save the indices to separate files
    with open(f'{path}/train_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, train_indices)))

    with open(f'{path}/val_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, val_indices)))

    with open(f'{path}/test_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, test_indices)))


if __name__ == '__main__':
    args = parse_args()

    project_root = args.project_root
    path = f'{project_root}/GraphCheck/dataset/extracted_KG'
    data_name = args.data_name
    
    dataset = pd.read_pickle(f'{path}/{data_name}/{data_name}.pkl')


    doc_kgs = dataset['doc_kg']
    claim_kgs = dataset['claim_kg']
    labels = dataset['label']

    step_one()
    step_two()
    generate_split(len(dataset), f'{path}/{data_name}/split')