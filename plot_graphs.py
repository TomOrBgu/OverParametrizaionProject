import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_and_clip_discrepancies(folder_path, disc_type, clip_min, clip_max):
    discrepancies = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(f'{disc_type}_discrepancies.npy'):
            model_name = filename.split('_')[0]
            data = np.load(os.path.join(folder_path, filename))
            clipped_data = np.clip(data, clip_min, clip_max)
            discrepancies[model_name] = clipped_data
    return discrepancies

def plot_combined_discrepancies(discrepancies, disc_type, output_folder):
    plt.figure(figsize=(12, 8))
    markers = ['o', 's', '^', 'D', 'v']

    for (name, values), marker in zip(discrepancies.items(), markers):
        epochs = range(1, len(values) + 1)
        plt.plot(epochs, values, label=name, marker=marker, markersize=6)

    plt.xlabel('Epochs')
    plt.ylabel(f'{disc_type.capitalize()} Discrepancy')
    plt.title(f'Clipped {disc_type.capitalize()} Discrepancy over Epochs for All ResNet Variants')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'all_models_clipped_{disc_type}_discrepancy.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot clipped discrepancies")
    parser.add_argument('--folder_path', type=str, help='Path to the folder containing .npy files',default='resnet_18_34_50_101_152_lr0.01_batch128_noise50_epochs200_optimizersgd_cifar10_20240819')
    args = parser.parse_args()

    # Load and clip discrepancies
    jsd_discrepancies = load_and_clip_discrepancies(args.folder_path, 'jsd', 0, 0.6)
    softmax_discrepancies = load_and_clip_discrepancies(args.folder_path, 'softmax', 0, 0.6)

    # Plot combined discrepancies
    plot_combined_discrepancies(jsd_discrepancies, 'jsd', args.folder_path)
    plot_combined_discrepancies(softmax_discrepancies, 'softmax', args.folder_path)

if __name__ == "__main__":
    main()