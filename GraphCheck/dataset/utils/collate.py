from torch_geometric.data import Batch


def collate_fn(original_batch):
    batch = {}
    for k in original_batch[0].keys():
        batch[k] = [d[k] for d in original_batch]
    if 'claim_kg' in batch:
        batch['claim_kg'] = Batch.from_data_list(batch['claim_kg'])
    if 'doc_kg' in batch:
        batch['doc_kg'] = Batch.from_data_list(batch['doc_kg'])
    return batch
