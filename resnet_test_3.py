import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os
from datetime import datetime
from discrepancy_metrics import compute_dispersion
import argparse
import random
def create_experiment_folder(model_names, args):
    current_date = datetime.now().strftime("%Y%m%d")
    model_names_str = "_".join([model.replace("ResNet", "") for model in model_names])
    folder_name = f"resnet_{model_names_str}_lr{args.lr}_batch{args.batch_size}_noise{int(args.noise_level*100)}_epochs{args.epochs}_optimizer{args.optimizer}_{args.dataset}_{current_date}_noisyweights"
    os.makedirs(folder_name, exist_ok=True)
    return folder_name

def add_label_noise(dataset, noise_level=0.2, num_classes=10):
    num_samples = len(dataset)
    num_noisy_labels = int(noise_level * num_samples)

    indices = np.random.choice(num_samples, num_noisy_labels, replace=False)
    noisy_labels = np.random.randint(0, num_classes, num_noisy_labels)

    for idx, noisy_label in zip(indices, noisy_labels):
        dataset.targets[idx] = noisy_label
    return dataset

def initialize_with_noise(net1, net2, noise_scale=0.001):
    with torch.no_grad():
        for param1, param2 in zip(net1.parameters(), net2.parameters()):
            noise = (torch.rand_like(param1.data) * 2 - 1) * noise_scale  # Uniformly distributed between [-1, 1], then scaled to [-noise_scale, noise_scale]
            param2.data = param1.data + noise

            # check if the weights are the same
            print(torch.allclose(param1.data, param2.data, atol=1e-13))

            # check if weights are the same up to 0.00001 
            print(torch.allclose(param1.data, param2.data, atol=1e-8))
            


def train_and_evaluate(model_func, args):
    num_classes = 10 if args.dataset in ['cifar10', 'mnist'] else 100
    net1 = model_func(num_classes=num_classes)
    net2 = model_func(num_classes=num_classes)

    # Initialize net2 with weights similar to net1 but with small noise
    initialize_with_noise(net1, net2, noise_scale=0.00000000001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net1 = net1.to(device)
    net2 = net2.to(device)

    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()

    if args.optimizer.lower() == 'sgd':
        optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'adam':
        optimizer1 = optim.Adam(net1.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer2 = optim.Adam(net2.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    if args.dataset == 'cifar100':
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            dataset_class = torchvision.datasets.CIFAR100
    elif args.dataset == 'cifar10':
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])
            dataset_class = torchvision.datasets.CIFAR10
    elif args.dataset == 'mnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            dataset_class = torchvision.datasets.MNIST
    else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")

    trainset = dataset_class(root='./data', train=True, download=True, transform=transform)
    trainset = add_label_noise(trainset, noise_level=args.noise_level, num_classes=num_classes)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    softmax_discrepancies = []
    classification_discrepancies = []
    jsd_discrepancies = []
    outputs_discrepancies = []
    losses1 = []
    losses2 = []

    for epoch in range(args.epochs):
        net1.train()
        net2.train()
        loss1_total = 0
        loss2_total = 0
        for data in trainloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            
            optimizer1.zero_grad()
            outputs1 = net1(inputs)
            loss1 = criterion1(outputs1, labels)
            loss1.backward()
            optimizer1.step()
            
            optimizer2.zero_grad()
            outputs2 = net2(inputs)
            loss2 = criterion2(outputs2, labels)
            loss2.backward()
            optimizer2.step()

            loss1_total += loss1.item()
            loss2_total += loss2.item()

        avg_loss1 = loss1_total / len(trainloader)
        avg_loss2 = loss2_total / len(trainloader)
        losses1.append(avg_loss1)
        losses2.append(avg_loss2)

        classification_discrepancy, jsd_discrepancy, outputs_discrepancy, softmax_discrepancy = compute_dispersion(net1, net2, trainloader,print_last=True)

        softmax_discrepancies.append(softmax_discrepancy)
        classification_discrepancies.append(classification_discrepancy)
        jsd_discrepancies.append(jsd_discrepancy)
        outputs_discrepancies.append(outputs_discrepancy)
      
        print(f"Epoch {epoch+1}/{args.epochs}, Model {model_func.__name__}, Classification Discrepancy: {classification_discrepancy:.4f}, JSD: {jsd_discrepancy:.4f}, Outputs Discrepancy: {outputs_discrepancy:.4f}, Softmax Discrepancy: {softmax_discrepancy:.4f}, Loss1: {avg_loss1:.4f}, Loss2: {avg_loss2:.4f}")

    return classification_discrepancies, jsd_discrepancies, outputs_discrepancies, softmax_discrepancies, losses1, losses2

def plot_discrepancies(model_name, softmax_disc, classification_disc, jsd_disc, outputs_disc, folder):
    discrepancies = {
        'Softmax': softmax_disc,
        'Classification': classification_disc,
        'JSD': jsd_disc,
        'Outputs': outputs_disc
    }

    for disc_name, disc_values in discrepancies.items():
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(disc_values) + 1)
        plt.plot(epochs, disc_values, label=model_name, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel(f'{disc_name} Discrepancy')
        plt.title(f'{disc_name} Discrepancy over Epochs for {model_name}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(folder, f'{model_name}_{disc_name.lower()}_discrepancy.png'))
        plt.close()

def plot_combined_discrepancies(results, folder):
    discrepancy_types = ['softmax', 'classification', 'jsd', 'outputs']
    markers = ['o', 's', '^', 'D', 'v']

    for disc_type in discrepancy_types:
        plt.figure(figsize=(12, 8))
        for (name, discrepancies), marker in zip(results[disc_type], markers):
            epochs = range(1, len(discrepancies) + 1)
            plt.plot(epochs, discrepancies, label=name, marker=marker, markersize=6)

        plt.xlabel('Epochs')
        plt.ylabel(f'{disc_type.capitalize()} Discrepancy')
        plt.title(f'{disc_type.capitalize()} Discrepancy over Epochs for All ResNet Variants')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(folder, f'all_models_{disc_type}_discrepancy.png'))
        plt.close()

def plot_losses(model_name, losses1, losses2, folder):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(losses1) + 1)
    plt.plot(epochs, losses1, label=f'{model_name}_1', marker='o')
    plt.plot(epochs, losses2, label=f'{model_name}_2', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training Losses for {model_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f'{model_name}_losses.png'))
    plt.close()

def plot_combined_losses(results, folder):
    plt.figure(figsize=(12, 8))
    markers = ['o', 's', '^', 'D', 'v']
    colors = ['b', 'g', 'r', 'c', 'm']

    for (name, losses1, losses2), marker, color in zip(results['losses'], markers, colors):
        epochs = range(1, len(losses1) + 1)
        plt.plot(epochs, losses1, label=f'{name}_1', marker=marker, color=color, linestyle='-')
        plt.plot(epochs, losses2, label=f'{name}_2', marker=marker, color=color, linestyle='--')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Losses for All ResNet Variants')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, 'all_models_losses.png'))
    plt.close()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description="ResNet Discrepancy Analysis")
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'], help='optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum (for SGD)')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'mnist'], help='dataset')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--noise_level', type=float, default=0.3, help='label noise level')
    parser.add_argument('--models', nargs='+', default=['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152'], 
                        help='ResNet models to train')

    args = parser.parse_args()


  

    # Call this function at the start of your main() function
    set_seed(42)  # You can use any integer as the seed

    resnet_configs = {
        'ResNet18': models.resnet18,
        'ResNet34': models.resnet34,
        'ResNet50': models.resnet50,
        'ResNet101': models.resnet101,
        'ResNet152': models.resnet152
    }

    # Filter resnet_configs based on the models specified in args
    resnet_configs = {k: v for k, v in resnet_configs.items() if k in args.models}

    experiment_folder = create_experiment_folder(list(resnet_configs.keys()), args)
    # save the arguments in a text file
    with open(os.path.join(experiment_folder, 'args.txt'), 'w') as f:
        f.write(str(args))

    results = {
        'softmax': [],
        'classification': [],
        'jsd': [],
        'outputs': [],
        'losses': []
    }

    for name, model_func in resnet_configs.items():
        print(f"Training {name}...")
        classification_discrepancies, jsd_discrepancies, outputs_discrepancies, softmax_discrepancies, losses1, losses2 = train_and_evaluate(model_func, args)
        
        results['softmax'].append((name, softmax_discrepancies))
        results['classification'].append((name, classification_discrepancies))
        results['jsd'].append((name, jsd_discrepancies))
        results['outputs'].append((name, outputs_discrepancies))
        results['losses'].append((name, losses1, losses2))

        np.save(os.path.join(experiment_folder, f'{name}_softmax_discrepancies.npy'), np.array(softmax_discrepancies))
        np.save(os.path.join(experiment_folder, f'{name}_classification_discrepancies.npy'), np.array(classification_discrepancies))
        np.save(os.path.join(experiment_folder, f'{name}_jsd_discrepancies.npy'), np.array(jsd_discrepancies))
        np.save(os.path.join(experiment_folder, f'{name}_outputs_discrepancies.npy'), np.array(outputs_discrepancies))
        np.save(os.path.join(experiment_folder, f'{name}_losses.npy'), np.array([losses1, losses2]))

        plot_discrepancies(name, softmax_discrepancies, classification_discrepancies, jsd_discrepancies, outputs_discrepancies, experiment_folder)
        plot_losses(name, losses1, losses2, experiment_folder)

    plot_combined_discrepancies(results, experiment_folder)
    plot_combined_losses(results, experiment_folder)

if __name__ == "__main__":
    main()