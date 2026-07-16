"""
Data Utilities for Federated Multimodal Learning

Provides synthetic data generation and dataset handling.
"""

import torch
import torchvision
from torchvision import transforms
from typing import Dict, List, Any, Tuple
import numpy as np


class MultimodalDataset(torch.utils.data.Dataset):
    """Multimodal dataset wrapper."""
    
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]


def generate_synthetic_multimodal_data(num_samples: int = 1000, 
                                       num_classes: int = 10) -> List[Dict[str, Any]]:
    """
    Generate synthetic multimodal data.
    
    Args:
        num_samples: Number of samples to generate
        num_classes: Number of classes
    
    Returns:
        List of multimodal data samples
    """
    data = []
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    mnist_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    
    for i in range(min(num_samples, len(mnist_dataset))):
        image, label = mnist_dataset[i]
        
        text_desc = generate_text_description(image, label)
        
        tabular_stats = compute_tabular_features(image)
        
        data.append({
            'modalities': {
                'image': image,
                'text': text_desc,
                'tabular': tabular_stats
            },
            'label': label
        })
    
    return data


def generate_text_description(image: torch.Tensor, label: int) -> str:
    """Generate text description from image."""
    descriptions = {
        0: "zero, oval shape, centered",
        1: "one, vertical line, tall",
        2: "two, curved line, descending",
        3: "three, two curves, symmetrical",
        4: "four, angular, closed top",
        5: "five, curved bottom, open top",
        6: "six, circular, with tail",
        7: "seven, diagonal line, ascending",
        8: "eight, two loops, stacked",
        9: "nine, circular top, curved bottom"
    }
    
    pixel_desc = []
    image_np = image.squeeze().numpy()
    
    if np.mean(image_np) > 0.5:
        pixel_desc.append("bright")
    else:
        pixel_desc.append("dark")
    
    if np.std(image_np) > 0.2:
        pixel_desc.append("high contrast")
    else:
        pixel_desc.append("low contrast")
    
    return f"{descriptions[label]}, {', '.join(pixel_desc)}"


def compute_tabular_features(image: torch.Tensor) -> List[float]:
    """Compute tabular features from image."""
    image_np = image.squeeze().numpy()
    
    features = [
        float(np.mean(image_np)),
        float(np.std(image_np)),
        float(np.max(image_np)),
        float(np.min(image_np)),
        float(np.median(image_np)),
        float(np.sum(image_np > 0.5) / image_np.size),
        float(np.sum(image_np < 0.1) / image_np.size),
        float(np.mean(np.abs(np.diff(image_np, axis=0)))),
        float(np.mean(np.abs(np.diff(image_np, axis=1)))),
        float(np.sqrt(np.sum(image_np ** 2)))
    ]
    
    return features


def split_modalities_across_clients(data: List[Dict[str, Any]], 
                                    num_clients: int = 3,
                                    missing_modalities_per_client: int = 1) -> List[List[Dict[str, Any]]]:
    """
    Split multimodal data across clients with random missing modalities.
    
    Args:
        data: Original multimodal data
        num_clients: Number of clients
        missing_modalities_per_client: Number of modalities each client should miss
    
    Returns:
        List of client datasets
    """
    modalities = ['image', 'text', 'tabular']
    client_datasets = []
    
    for client_id in range(num_clients):
        client_data = []
        
        np.random.seed(client_id)
        missing = np.random.choice(modalities, size=missing_modalities_per_client, replace=False)
        
        for sample in data:
            filtered_modalities = {}
            
            for modality in modalities:
                if modality not in missing:
                    filtered_modalities[modality] = sample['modalities'][modality]
            
            client_data.append({
                'modalities': filtered_modalities,
                'label': sample['label']
            })
        
        client_datasets.append(client_data)
    
    return client_datasets


def create_dataloader(dataset: List[Dict[str, Any]], batch_size: int = 32, 
                      shuffle: bool = True) -> torch.utils.data.DataLoader:
    """Create dataloader for multimodal data."""
    multimodal_dataset = MultimodalDataset(dataset)
    
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        modalities = {}
        labels = []
        
        for sample in batch:
            labels.append(sample['label'])
            
            for modality, data in sample['modalities'].items():
                if modality not in modalities:
                    modalities[modality] = []
                modalities[modality].append(data)
        
        for modality in modalities:
            if isinstance(modalities[modality][0], torch.Tensor):
                modalities[modality] = torch.stack(modalities[modality])
            elif isinstance(modalities[modality][0], str):
                modalities[modality] = modalities[modality]
            else:
                modalities[modality] = torch.tensor(modalities[modality])
        
        return {
            'modalities': modalities,
            'labels': torch.tensor(labels)
        }
    
    return torch.utils.data.DataLoader(
        multimodal_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )


def generate_modality_mask(num_modalities: int = 3, mask_prob: float = 0.1) -> List[bool]:
    """Generate random modality mask."""
    return [torch.rand(1).item() < mask_prob for _ in range(num_modalities)]


def compute_mutual_information(feature1: torch.Tensor, feature2: torch.Tensor) -> float:
    """Compute mutual information between two feature sets."""
    p1 = F.softmax(feature1, dim=1)
    p2 = F.softmax(feature2, dim=1)
    
    mi = 0.0
    for i in range(feature1.size(0)):
        for j in range(feature1.size(1)):
            if p1[i, j] > 0 and p2[i, j] > 0:
                mi += p1[i, j] * torch.log(p2[i, j] / p1[i, j])
    
    return float(mi)


if __name__ == "__main__":
    print("Testing multimodal data utilities...")
    
    data = generate_synthetic_multimodal_data(num_samples=100)
    print(f"Generated {len(data)} samples")
    
    client_data = split_modalities_across_clients(data, num_clients=3)
    for i, cd in enumerate(client_data):
        modalities = set()
        for sample in cd:
            modalities.update(sample['modalities'].keys())
        print(f"Client {i}: {len(cd)} samples, modalities: {modalities}")
    
    dataloader = create_dataloader(client_data[0], batch_size=8)
    for batch in dataloader:
        print(f"Batch labels shape: {batch['labels'].shape}")
        for modality, data in batch['modalities'].items():
            print(f"  {modality}: {data.shape if hasattr(data, 'shape') else len(data)}")
        break
    
    print(" Data utilities test completed!")