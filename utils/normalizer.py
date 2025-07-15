from torch_geometric.transforms import NormalizeScale

def normalize(ds):
    normalize_transform = NormalizeScale()
    normalized_ds = []
    
    for _, data in enumerate(ds):
        normalized_data = normalize_transform(data)
        normalized_ds.append(normalized_data)

    return normalized_ds