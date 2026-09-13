import torch
import torch.nn as nn
import numpy as np
from scipy.stats import entropy
import yaml
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
from collections import defaultdict

# Define the MNIST classifier
class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class FMNISTClassifier(nn.Module):
    def __init__(self):
        super(FMNISTClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class KMNISTClassifier(nn.Module):
    def __init__(self):
        super(KMNISTClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class NotMNISTClassifier(nn.Module):
    def __init__(self):
        super(NotMNISTClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#calculate the Image generation score - similar to inception score , but with dataspecific classifier
def calculate_image_score(mnist_classifier,client_gen1,client_gen2,server_GEN,device,server_gen_indeces,Generator_extra_layers,num_samples=5000,type='mnist'):

    with open("./configs.yaml", "r") as f:
        config = yaml.safe_load(f)

    training = config["training"]
    batch_size = training["batch_size"]

    client_gen1.eval()
    client_gen2.eval()
    server_GEN.eval()
    mnist_classifier.eval()
    all_preds=[]
    start=0
    if type=='mnist' or type=='fmnist' or type == 'kmnist' or type == 'notmnist':
        num_classes=10
    else:
        num_classes=None
    # fake_images = []
    with torch.no_grad():

        for _ in range(num_samples // batch_size):
            im_labels = torch.randint(start, num_classes, (batch_size,)).to(device)
            z = torch.randn(batch_size, 100).to(device)
            c1=client_gen1(z,im_labels)
            c2=server_GEN.forward_c(c1,[client_gen1.additional_layers_count - server_gen_indeces[0] , len(Generator_extra_layers) -1*client_gen2.additional_layers_count -server_gen_indeces[0] -1 ])
            generated = client_gen2(c2).detach()
            with torch.no_grad():
                preds=torch.nn.functional.softmax(mnist_classifier(generated),dim=1)
                all_preds.append(preds.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)

    # Compute marginal p(y)
    p_y = np.mean(all_preds, axis=0)

    kl_divs = []
    for pred in all_preds:
        kl_div = entropy(pred, p_y)  # D_KL(p(y|x) || p(y))
        kl_divs.append(kl_div)

    # Calculate Inception Score
    mnist_score = np.exp(np.mean(kl_divs))

    client_gen1.train()
    client_gen2.train()
    server_GEN.train()
    print(f"{type} Score: {mnist_score}")
    return mnist_score

#generate fake data to be trained on by the classifier
def generate_classifier_data(client_gen1,client_gen2,server_GEN,device,server_gen_indeces,Generator_extra_layers,num_samples=20000,type='mnist'):
    client_gen1.eval()
    client_gen2.eval()
    server_GEN.eval()
    with open("./configs.yaml", "r") as f:
        config = yaml.safe_load(f)

    training = config["training"]
    batch_size = training["batch_size"]
    start=0
    if type =='mnist' or type =='fmnist' or type=='kmnist' or type=='notmnist':
        num_classes=10

    #counter for images
    k=[0]*num_classes
    all_images = []
    all_labels = []
    # fake_images = []
    with torch.no_grad():

        for _ in range(num_samples // batch_size):
            im_labels = torch.randint(start, num_classes, (batch_size,)).to(device)
            z = torch.randn(batch_size, 100).to(device)
            c1=client_gen1(z,im_labels)
            c2=server_GEN.forward_c(c1,[client_gen1.additional_layers_count - server_gen_indeces[0] , len(Generator_extra_layers) -1*client_gen2.additional_layers_count -server_gen_indeces[0] -1 ])
            generated = client_gen2(c2).detach()
            all_images.append(generated.cpu())
            all_labels.append(im_labels.cpu())



    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    with open("./configs.yaml", "r") as f:
        config = yaml.safe_load(f)

    training = config["training"]
    experiment = config["experiment"]
    batch_size = training["batch_size"]
    scenario = experiment["scenario"]
    # Save both as a tuple
    save_path = f"./synthetic_data/HuSCFGAN/scenario_{scenario}_{type}.pt"
    torch.save((all_images, all_labels), save_path)
    print(f"Saved {len(all_images)} synthetic samples to: {save_path}")
    client_gen1.train()
    client_gen2.train()
    server_GEN.train()
    return

#load pre-trained classifier on the datasets
def get_dataset_classifiers(device):
    mnist_classifier = MNISTClassifier().to(device)
    mnist_classifier.load_state_dict(torch.load('./Metrics/mnist_classifier.pth',weights_only=False))
    fmnist_classifier = FMNISTClassifier().to(device)
    fmnist_classifier.load_state_dict(torch.load('./Metrics/fmnist_classifier.pth',weights_only=False))
    kmnist_classifier=KMNISTClassifier().to(device)
    kmnist_classifier.load_state_dict(torch.load('./Metrics/kmnist_classifier.pth',weights_only=False))
    notmnist_classifier=NotMNISTClassifier().to(device)
    notmnist_classifier.load_state_dict(torch.load('./Metrics/notmnist_classifier.pth',weights_only=False))
    return mnist_classifier,fmnist_classifier,kmnist_classifier,notmnist_classifier

class Classifier(nn.Module):
    def __init__(self,num_classes):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Adjust based on image size
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
#This function return the accuracy per label
def accuracy_per_label(all_labels, all_preds, label_names=None):
    correct_per_class = defaultdict(int)
    total_per_class = defaultdict(int)

    for true, pred in zip(all_labels, all_preds):
        total_per_class[true] += 1
        if true == pred:
            correct_per_class[true] += 1

    all_classes = sorted(set(all_labels))
    print("Per-label accuracy:\n")
    for cls in all_classes:
        acc = 100 * correct_per_class[cls] / total_per_class[cls]
        label_str = f"Class {cls}" if not label_names else label_names[cls]
        print(f"{label_str:<10}: {acc:.2f}% ({correct_per_class[cls]}/{total_per_class[cls]})")

#this function returns the classification metrics like accuracy and others of the classifier trained on 30K Fake Data
def get_accuracy(test_dataloader,type='mnist'):

    with open("./configs.yaml", "r") as f:
        config = yaml.safe_load(f)

    experiment = config["experiment"]
    scenario = experiment["scenario"]
    if type =='mnist' or type =='fmnist' or type=='kmnist' or type=='notmnist':
        num_classes=10
    data_path = f"./synthetic_data/HuSCFGAN/scenario_{scenario}_{type}.pt"
    images, labels = torch.load(data_path,weights_only=False)
    dataset = torch.utils.data.TensorDataset(images, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    classifier = Classifier(num_classes).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)

    print('--------------------------------------')
    print('Training Classifier on 30k Synthetic Data: ')
    # Training loop
    for epoch in range(15):
        for images, labels in dataloader:
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = classifier(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/15], Loss: {loss.item():.4f}")


    # Compute accuracy
    all_preds = []
    all_labels = []
    total = 0
    correct = 0

    classifier.eval()
    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.cuda(), labels.cuda()
            outputs = classifier(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    # Compute macro FPR: false positives / total actual negatives (per class, then averaged)
    cm = confusion_matrix(all_labels, all_preds)
    num_classes = cm.shape[0]
    fpr_list = []

    for i in range(num_classes):
        fp = cm[:, i].sum() - cm[i, i]
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
        fpr_list.append(fpr)
    fpr_macro = np.mean(fpr_list)

    # accuracy_per_label(all_labels, all_preds)
    return accuracy,100*precision,100*recall,100*f1,100*fpr_macro


