import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Subset, random_split, Dataset
from torchvision import transforms, datasets
from torchvision.transforms.functional import to_pil_image
import yaml

class TransformedPTDataset(Dataset):
    def __init__(self, pt_file, transform=None):
        data = torch.load(pt_file,weights_only=False)
        self.images = data['images']  # shape: [N, 1, 28, 28]
        self.labels = data['labels']  # shape: [N]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = transforms.ToPILImage()(image)  # <- Correct: call the transform here
            image = self.transform(image)
        return image, label

# Transforms
mnist_fashion_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])




def generate_exclusion_dict(num_clients,scenario=1, label_range=10, seed=42,offset=0):
    random.seed(seed)

    if scenario==1 or scenario ==3 or scenario ==6 :
        exclusion_counts = [0, 0, 0, 0]
        probabilities = [0.25, 0.3, 0.1, 0.1]  # Matches the original approximate ratios
    elif scenario ==2 or scenario ==4:
        exclusion_counts = [0, 2, 3, 4]
        probabilities = [0.25, 0.3, 0.1, 0.1]
    elif scenario==5:
        exclusion_counts = [2, 3]
        probabilities = [0.4, 0.6]

    the_exclusions = {}
    for i in range(num_clients):
        # Choose how many exclusions this client will have
        exclusion_count = random.choices(exclusion_counts, weights=probabilities, k=1)[0]
        # Randomly choose unique labels to exclude
        exclusions = random.sample(range(label_range), exclusion_count)
        the_exclusions[i+offset] = exclusions

    return the_exclusions




def generate_client_sizes(num_clients,scenario=1,total_dataset_size=60000, seed=42):
    random.seed(seed)
    size_per_client=total_dataset_size//num_clients
    client_max_samples = {}
    if scenario ==1 or scenario ==3 or scenario ==6:
        for i in range(num_clients):
            client_max_samples[i]=size_per_client
    elif scenario ==2 or scenario ==4:
        smaller_ratio=0.6666667
        bigger_ratio=1
        for i in range(num_clients):
            r=random.random()
            if r<0.3:
                client_max_samples[i]=int(smaller_ratio*size_per_client)
            else:
                client_max_samples[i]=int(bigger_ratio*size_per_client)
    elif scenario ==5:
        for i in range(num_clients):
            r=random.random()
            smaller_ratio=0.1666667
            medium_ratio=0.333333
            bigger_ratio=1
            if r<0.2:
                client_max_samples[i]=int(smaller_ratio*size_per_client)
            elif r>=0.2 and r<0.5:
                client_max_samples[i]=int(medium_ratio*size_per_client)
            else:
                client_max_samples[i]=int(bigger_ratio*size_per_client)
    return client_max_samples

# Helper Functions
def split_dataset_equally(dataset, num_parts):
    sizes = [len(dataset) // num_parts] * num_parts
    sizes[0] += len(dataset) % num_parts
    return random_split(dataset, sizes)

def filter_dataset_by_class(dataset, excluded_classes):
    indices = [idx for idx, (_, label) in enumerate(dataset) if label not in excluded_classes]
    return Subset(dataset, indices)

def limit_subset(dataset, max_samples):
    if len(dataset) > max_samples:
        indices = np.random.choice(len(dataset), max_samples, replace=False)
        return Subset(dataset, indices)
    return dataset


def get_data_loaders():
    # Parameters

    # Load YAML settings
    with open("./configs.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Extract values from the nested structure
    clients = config["clients"]
    training = config["training"]
    experiment = config["experiment"]

    # Assign to variables
    num_digit_clients = clients["num_digit_clients"]
    num_fashion_clients = clients["num_fashion_clients"]
    num_kmnist_clients = clients["num_kmnist_clients"]
    num_notmnist_clients = clients["num_notmnist_clients"]

    batch_size = training["batch_size"]
    scenario = experiment["scenario"]
    num_clients = num_digit_clients + num_fashion_clients +num_kmnist_clients + num_notmnist_clients 
    test_loaders={}
    
    digit_exclusions={}
    fashion_exclusions={}
    letter_exclusions={}
    kmnist_exclusions={}
    
    # # Load Datasets
    if num_digit_clients>0:
        mnist_digits= datasets.MNIST(root="./Data", train=True, download=True, transform=mnist_fashion_transform)
        real_test_dataset = datasets.MNIST(root="./Data", train=False, transform=mnist_fashion_transform, download=True)
        mnist_test = torch.utils.data.DataLoader(real_test_dataset, batch_size=batch_size, shuffle=False)
        test_loaders['mnist']=mnist_test
        digit_exclusions=generate_exclusion_dict(num_digit_clients,scenario=scenario, label_range=10, seed=42)
    if num_fashion_clients>0:
        fmnist_data = datasets.FashionMNIST(root="./Data", train=True, download=True, transform=mnist_fashion_transform)
        real_test_dataset = datasets.FashionMNIST(root="./Data", train=False, transform=mnist_fashion_transform, download=True)
        fmnist_test = torch.utils.data.DataLoader(real_test_dataset, batch_size=batch_size, shuffle=False)
        test_loaders['fmnist']=fmnist_test
        fashion_exclusions=generate_exclusion_dict(num_fashion_clients,scenario=scenario, label_range=10, seed=32,offset=num_digit_clients)
    if num_kmnist_clients>0:
        kmnist_data=datasets.KMNIST(root="./Data", train=True, download=True, transform=mnist_fashion_transform)
        real_test_dataset = datasets.KMNIST(root="./Data", train=False, transform=mnist_fashion_transform, download=True)
        kmnist_test = torch.utils.data.DataLoader(real_test_dataset, batch_size=batch_size, shuffle=False)
        test_loaders['kmnist']=kmnist_test
        kmnist_exclusions=generate_exclusion_dict(num_kmnist_clients,scenario=scenario, label_range=10, seed=32,offset=num_digit_clients+num_fashion_clients)
    if num_notmnist_clients>0:
    # Load training and test datasets using PT file
        data_path = f"./Data/NotMNIST.pt"
        full_notmnist_dataset = TransformedPTDataset(data_path,transform=mnist_fashion_transform)
        train_size = int(0.8 * len(full_notmnist_dataset))
        test_size = len(full_notmnist_dataset) - train_size
        # Split
        notmnist_dataset, notmnist_test_dataset = random_split(full_notmnist_dataset, [train_size, test_size])
        notmnist_test = torch.utils.data.DataLoader(notmnist_test_dataset, batch_size=batch_size, shuffle=False)
        test_loaders['notmnist']=notmnist_test
        letter_exclusions=generate_exclusion_dict(num_notmnist_clients,scenario=scenario, label_range=10, seed=32,offset=num_digit_clients+num_fashion_clients+num_kmnist_clients)

    

    digit_client_datasets = []
    client_max_samples=generate_client_sizes(num_clients,scenario=scenario)

    if num_digit_clients > 0:
        digit_splits = split_dataset_equally(mnist_digits, num_digit_clients)
        for client_id in range(num_digit_clients):
            dataset = digit_splits[client_id]
            exclusions = digit_exclusions.get(client_id, [])
            filtered = filter_dataset_by_class(dataset, exclusions)
            limited = limit_subset(filtered, client_max_samples[client_id])
            digit_client_datasets.append(limited)

    # Fashion Clients
    fashion_client_datasets = []
    if num_fashion_clients > 0:
        fashion_splits = split_dataset_equally(fmnist_data, num_fashion_clients)
        for client_offset in range(num_fashion_clients):
            client_id = num_digit_clients + client_offset
            exclusions = fashion_exclusions.get(client_id, [])
            dataset = fashion_splits[client_offset]
            filtered = filter_dataset_by_class(dataset, exclusions)
            limited = limit_subset(filtered, client_max_samples[client_id])
            fashion_client_datasets.append(limited)

    kmnist_client_datasets = []
    if num_kmnist_clients > 0:
        kmnist_splits = split_dataset_equally(kmnist_data, num_kmnist_clients)
        for client_offset in range(num_kmnist_clients):
            client_id = num_kmnist_clients + client_offset
            exclusions = kmnist_exclusions.get(client_id, [])
            dataset = kmnist_splits[client_offset]
            filtered = filter_dataset_by_class(dataset, exclusions)
            limited = limit_subset(filtered, client_max_samples[client_id])
            kmnist_client_datasets.append(limited)

    # Capital Letter Clients
    notmnist_client_datasets = []
    if num_notmnist_clients > 0:
        capital_splits = split_dataset_equally(notmnist_dataset, num_notmnist_clients)
        for client_offset in range(num_notmnist_clients):
            client_id = num_digit_clients + num_fashion_clients + client_offset
            exclusions = letter_exclusions.get(client_id, [])
            dataset = capital_splits[client_offset]
            filtered = filter_dataset_by_class(dataset, exclusions)
            limited = limit_subset(filtered, client_max_samples[client_id])
            notmnist_client_datasets.append(limited)


    # Combine All Clients
    all_client_datasets = digit_client_datasets + fashion_client_datasets +kmnist_client_datasets+ notmnist_client_datasets
    clients_data_loaders = [
        DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for dataset in all_client_datasets
    ]
    client_sizes = [len(dataset) for dataset in all_client_datasets]
    return clients_data_loaders,client_sizes,test_loaders,digit_exclusions,fashion_exclusions,kmnist_exclusions,letter_exclusions


