import argparse
import torch
import torchvision
from torchvision import datasets, transforms
import numpy as np
import torch.optim as optim
import os
import matplotlib.pyplot as plt     

from discrepancy_metrics import compute_dispersion , moving_average

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Introduce 20% label noise
def add_label_noise(dataset, noise_level=0.2):
    num_samples = len(dataset)
    num_noisy_labels = int(noise_level * num_samples)

    indices = np.random.choice(num_samples, num_noisy_labels, replace=False)
    noisy_labels = np.random.randint(0, 10, num_noisy_labels)

    for idx, noisy_label in zip(indices, noisy_labels):
        dataset.targets[idx] = noisy_label

def train(net, optimizer, epoch, trainloader,criterion):
    net.train()
    running_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(trainloader)
    print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')
    return avg_loss

def get_model_from_torchvision(model_name):
    if model_name == 'resnet18':
        model = torchvision.models.resnet18(num_classes=10)
    elif model_name == 'resnet34':
        model = torchvision.models.resnet34(num_classes=10)
    elif model_name == 'resnet50':
        model = torchvision.models.resnet50(num_classes=10)
    elif model_name == 'resnet101':
        model = torchvision.models.resnet101(num_classes=10)
    elif model_name == 'resnet152':
        model = torchvision.models.resnet152(num_classes=10)
    else:
        raise ValueError('Invalid model name')
    return model




def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a ResNet model on CIFAR-10')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='input batch size for training (default: 100)')
    parser.add_argument('--epochs', type=int, default=30,)
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--model_name', type=str, default='resnet18',choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
    parser.add_argument('--noise_level', type=float, default=0.2)


    args = parser.parse_args()
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    momentum = args.momentum
    weight_decay = args.weight_decay
    noise_level = args.noise_level
    model_name = args.model_name

    EXPIREMENT_NAME = f'{model_name}_noise_{noise_level}_epochs_{epochs}_lr_{lr}_momentum_{momentum}_weight_decay_{weight_decay}'
    os.makedirs(EXPIREMENT_NAME, exist_ok=True)

    # Load CIFAR-10 dataset
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Add label noise
    add_label_noise(trainset, noise_level=noise_level)

    net1 = get_model_from_torchvision(model_name)
    net1 = net1.to(device)

    net2 = get_model_from_torchvision(model_name)
    net2 = net2.to(device)


    optimizer1 = optim.SGD(net1.parameters(), lr=lr, momentum=momentum,weight_decay=weight_decay)
    optimizer2 = optim.SGD(net2.parameters(), lr=lr, momentum=momentum,weight_decay=weight_decay)

    criterion = torch.nn.CrossEntropyLoss()

    # Train both networks and compute discrepancies
    classification_discrepancies = []
    weight_discrepancies = []
    outputs_discrepancies = []

    jsd_discrepancies = []

    losses1 = []
    losses2 = []

    for epoch in range(epochs):
        print(f"--- Epoch {epoch} ---")
        # discrepancy = compute_classification_discrepancy(net1, net2, trainloader) # ORIGINAL FROM PAPER
        # discrepancy = discrepancy.detach().cpu().numpy()
        # discrepancies.append(discrepancy)

        classification_discrepancy , jsd_discrepancy , outputs_discrepancy = compute_dispersion(net1, net2, trainloader)  # ORIGINAL FROM PAPER



        classification_discrepancies.append(classification_discrepancy)
        # weight_discrepancies.append(weight_discrepancy)
        jsd_discrepancies.append(jsd_discrepancy)
        outputs_discrepancies.append(outputs_discrepancy)

        print(f'Epoch {epoch}, Classification Discrepancy: {classification_discrepancy:.4f}, 'f'JSD: {jsd_discrepancy:.4f},' f'outputs_discrepancy:{outputs_discrepancy}' )



        #print(f'Epoch {epoch+1}, Discrepancy: {discrepancy:.4f}')

        loss1 = train(net1, optimizer1, epoch, trainloader,criterion)
        loss2 = train(net2, optimizer2, epoch, trainloader,criterion)

        losses1.append(loss1)
        losses2.append(loss2)

    # Save the discrepancies to a file
    np.save(f'{EXPIREMENT_NAME}/l2_discrepancies.npy', outputs_discrepancies)
    np.save(f'{EXPIREMENT_NAME}/classification_discrepancies.npy', classification_discrepancies)
    np.save(f'{EXPIREMENT_NAME}weight_discrepancies.npy', weight_discrepancies) 

    # Save the figures
    plt.figure()
    plt.plot(epoch,moving_average(classification_discrepancies),marker='-')
    plt.xlabel('Epoch')
    plt.ylabel('Classification Discrepancy')
    plt.title('Classification Discrepancy')
    plt.savefig(f'{EXPIREMENT_NAME}/classification_discrepancy.png')

    plt.figure()
    plt.plot(epoch,moving_average(jsd_discrepancies),marker='-')
    plt.xlabel('Epoch')
    plt.ylabel('JSD Discrepancy')
    plt.title('JSD Discrepancy')
    plt.savefig(f'{EXPIREMENT_NAME}/jsd_discrepancy.png')

    plt.figure()
    plt.plot(epoch,moving_average(outputs_discrepancies),marker='-')
    plt.xlabel('Epoch')
    plt.ylabel('Outputs Discrepancy')
    plt.title('Outputs Discrepancy')
    plt.savefig(f'{EXPIREMENT_NAME}/outputs_discrepancy.png')


if __name__ == '__main__':
    main()