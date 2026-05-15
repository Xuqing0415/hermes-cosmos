"""
Non-IID Text Data Partitioning Module

Implements various strategies for creating Non-IID data distributions across clients.
"""

import numpy as np
from datasets import load_dataset, Dataset
from typing import List, Dict, Any
import random


def create_non_iid_text_data(dataset_name: str = "wikitext", 
                             subset: str = "wikitext-2-raw-v1",
                             num_clients: int = 10,
                             alpha: float = 0.5,
                             split: str = "train"):
    """
    Create Non-IID text data using Dirichlet distribution.
    
    Args:
        dataset_name: HuggingFace dataset name
        subset: Dataset subset
        num_clients: Number of clients
        alpha: Dirichlet concentration parameter (lower = more non-IID)
        split: Dataset split to use
        
    Returns:
        client_datasets: List of datasets for each client
    """
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, subset, split=split)
    
    # Group by document/topic
    documents = []
    current_doc = []
    
    for example in dataset:
        text = example['text']
        if text.strip():
            current_doc.append(text)
        else:
            if current_doc:
                documents.append('\n'.join(current_doc))
                current_doc = []
    
    if current_doc:
        documents.append('\n'.join(current_doc))
    
    num_docs = len(documents)
    print(f"Total documents: {num_docs}")
    
    # Assign documents to clients using Dirichlet distribution
    doc_indices = np.arange(num_docs)
    np.random.shuffle(doc_indices)
    
    # Create topic preferences using Dirichlet
    dirichlet_samples = np.random.dirichlet([alpha] * num_clients, num_docs)
    client_assignments = np.argmax(dirichlet_samples, axis=1)
    
    # Assign documents to clients
    client_docs = [[] for _ in range(num_clients)]
    for doc_idx, client_idx in enumerate(client_assignments):
        client_docs[client_idx].append(documents[doc_indices[doc_idx]])
    
    # Create HuggingFace Datasets for each client
    client_datasets = []
    for i in range(num_clients):
        client_dataset = Dataset.from_dict({
            'text': client_docs[i],
            'client_id': [i] * len(client_docs[i])
        })
        client_datasets.append(client_dataset)
        print(f"Client {i}: {len(client_docs[i])} documents")
    
    return client_datasets


def create_synthetic_non_iid_data(num_clients: int = 10, 
                                  samples_per_client: int = 100,
                                  alpha: float = 0.5):
    """
    Create synthetic Non-IID text data for testing.
    
    Args:
        num_clients: Number of clients
        samples_per_client: Number of samples per client
        alpha: Non-IIDness parameter (lower = more non-IID)
        
    Returns:
        client_datasets: List of datasets for each client
    """
    topics = [
        "technology", "sports", "politics", "entertainment", "science",
        "business", "health", "education", "environment", "history"
    ]
    
    # Generate topic preferences
    topic_probs = np.random.dirichlet([alpha] * len(topics), num_clients)
    
    client_datasets = []
    
    for client_id in range(num_clients):
        samples = []
        for _ in range(samples_per_client):
            topic_idx = np.random.choice(len(topics), p=topic_probs[client_id])
            topic = topics[topic_idx]
            text = generate_topic_text(topic)
            samples.append({'text': text, 'topic': topic})
        
        client_dataset = Dataset.from_list(samples)
        client_datasets.append(client_dataset)
        
        # Show topic distribution for this client
        topic_counts = {}
        for s in samples:
            topic_counts[s['topic']] = topic_counts.get(s['topic'], 0) + 1
        print(f"Client {client_id} topic distribution: {topic_counts}")
    
    return client_datasets


def generate_topic_text(topic: str) -> str:
    """Generate synthetic text for a given topic."""
    templates = {
        "technology": [
            f"The latest {topic} trends include artificial intelligence and machine learning.",
            f"Advancements in {topic} are transforming industries worldwide.",
            f"Experts predict significant growth in {topic} sector next year.",
            f"New {topic} innovations are changing how we live and work."
        ],
        "sports": [
            f"The {topic} season has been exciting with many unexpected results.",
            f"Top athletes in {topic} are breaking records this year.",
            f"Fans are eagerly anticipating the upcoming {topic} championship.",
            f"Training techniques in {topic} have evolved significantly."
        ],
        "politics": [
            f"Recent {topic} developments have sparked nationwide debates.",
            f"Key {topic} leaders are meeting to discuss important issues.",
            f"Public opinion on {topic} matters continues to shift.",
            f"New {topic} policies are expected to be announced soon."
        ],
        "entertainment": [
            f"The {topic} industry is experiencing a renaissance with new releases.",
            f"Popular {topic} events are drawing large audiences.",
            f"Streaming platforms are changing {topic} consumption habits.",
            f"Talented artists are emerging in the {topic} scene."
        ],
        "science": [
            f"Groundbreaking {topic} discoveries are being made every day.",
            f"Researchers in {topic} are pushing the boundaries of knowledge.",
            f"New {topic} theories are challenging conventional wisdom.",
            f"Technological advances are accelerating {topic} progress."
        ],
        "business": [
            f"Global {topic} trends are shaping economic landscapes.",
            f"Entrepreneurs are finding new opportunities in {topic}.",
            f"Investors are showing interest in {topic} startups.",
            f"Market analysts predict strong {topic} growth."
        ],
        "health": [
            f"New {topic} research is improving patient outcomes.",
            f"Experts recommend focusing on {topic} and wellness.",
            f"Advances in {topic} care are saving lives.",
            f"Public {topic} awareness campaigns are gaining traction."
        ],
        "education": [
            f"Innovative {topic} methods are improving student outcomes.",
            f"Technology is transforming {topic} systems worldwide.",
            f"New {topic} policies are being implemented in schools.",
            f"Educators are adapting to changing {topic} needs."
        ],
        "environment": [
            f"Climate change is impacting {topic} globally.",
            f"Scientists are studying {topic} patterns and trends.",
            f"Conservation efforts are protecting {topic} resources.",
            f"Sustainable practices are becoming more important for {topic}."
        ],
        "history": [
            f"New {topic} discoveries are rewriting our understanding.",
            f"Historians are uncovering fascinating {topic} insights.",
            f"Ancient {topic} civilizations continue to amaze us.",
            f"Studying {topic} helps us understand the present."
        ]
    }
    
    return random.choice(templates.get(topic, templates["technology"]))


def tokenize_dataset(dataset, tokenizer, max_length: int = 512):
    """Tokenize a dataset for language modeling."""
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_overflowing_tokens=False
        )
    
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )
    
    # Format for causal language modeling
    tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    
    return tokenized


def split_train_test(client_datasets: List[Dataset], test_ratio: float = 0.1):
    """Split each client's dataset into train and test."""
    train_datasets = []
    test_datasets = []
    
    for client_dataset in client_datasets:
        split = client_dataset.train_test_split(test_size=test_ratio)
        train_datasets.append(split['train'])
        test_datasets.append(split['test'])
    
    return train_datasets, test_datasets


def create_iid_data(dataset_name: str = "wikitext", 
                    subset: str = "wikitext-2-raw-v1",
                    num_clients: int = 10,
                    split: str = "train"):
    """Create IID data distribution for comparison."""
    print(f"Creating IID data with {num_clients} clients")
    
    dataset = load_dataset(dataset_name, subset, split=split)
    shuffled = dataset.shuffle(seed=42)
    
    # Simple random split
    client_size = len(shuffled) // num_clients
    client_datasets = []
    
    for i in range(num_clients):
        start = i * client_size
        end = start + client_size if i < num_clients - 1 else len(shuffled)
        client_dataset = shuffled.select(range(start, end))
        client_datasets.append(client_dataset)
        print(f"Client {i}: {len(client_dataset)} examples")
    
    return client_datasets


if __name__ == "__main__":
    # Test Non-IID data creation
    print("=== Testing Non-IID Data Creation ===")
    client_datasets = create_synthetic_non_iid_data(num_clients=5, samples_per_client=50, alpha=0.3)
    print(f"\nCreated {len(client_datasets)} client datasets")
    
    # Test IID data creation
    print("\n=== Testing IID Data Creation ===")
    iid_datasets = create_iid_data(num_clients=3)
    print(f"Created {len(iid_datasets)} IID client datasets")
