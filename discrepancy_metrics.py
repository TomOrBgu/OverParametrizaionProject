import torch
import torch.nn.functional as F
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# To calculate the discrepancy between the two networks' outputs
def compute_classification_discrepancy(outputs1, outputs2):

    y1 = torch.argmax(outputs1,dim=1)
    y2 = torch.argmax(outputs2,dim=1)
    discrepancy = torch.sum(y1!=y2)

    return discrepancy
 


# # To calculate the weight discrepancy between the two networks
# def compute_weight_discrepancy(net1, net2):
#     total_discrepancy = 0.0
#     total_params = 0
#     for (param1, param2) in zip(net1.parameters(), net2.parameters()):
#         total_discrepancy += torch.sum((param1 - param2) ** 2).item()
#         total_params += param1.numel()
#     return (total_discrepancy / total_params) ** 0.5

# def compute_average_layerwise_weight_discrepancy(net1, net2):
#     total_discrepancy = 0.0
#     layer_count = 0

#     for (param1, param2) in zip(net1.parameters(), net2.parameters()):
#         # Compute the sum of weights for each layer
#         sum_weights1 = torch.sum(param1).item()
#         sum_weights2 = torch.sum(param2).item()

#         # Calculate the absolute difference in sums of weights for the layer
#         layer_discrepancy = abs(sum_weights1 - sum_weights2)

#         total_discrepancy += layer_discrepancy
#         layer_count += 1

#     # Compute the average discrepancy across all layers
#     average_discrepancy = total_discrepancy / layer_count
#     return average_discrepancy


def compute_jsd_discrepancy(output1, output2):

    epsilon = 1e-10  # Small constant to avoid log(0)

    outputs1 = F.softmax(output1, dim=1) + epsilon
    outputs2 = F.softmax(output2, dim=1) + epsilon

    # Re-normalize the distributions after adding epsilon
    outputs1 = outputs1 / torch.sum(outputs1, dim=1, keepdim=True)
    outputs2 = outputs2 / torch.sum(outputs2, dim=1, keepdim=True)

    # Calculate the Jensen-Shannon Divergence
    m = 0.5 * (outputs1 + outputs2)
    jsd = 0.5 * (F.kl_div(outputs1.log(), m, reduction='sum') + F.kl_div(outputs2.log(), m, reduction='sum'))
    return jsd.item()



# To calculate the discrepancy between the two networks' outputs
def compute_l2_discrepancy(outputs1, outputs2):

    return torch.sum(torch.norm(outputs1-outputs2,p=2,dim=1))


def compute_dispersion(net1, net2, loader):
    net1.eval()
    net2.eval()
    classifcation_discrepancy = 0.0
    jsd_dipcepancy = 0.0
    l2_discrepancy = 0.0
    num_samples = len(loader.dataset)
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(loader):
            inputs = inputs.to(device)
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)    
            classifcation_discrepancy += compute_classification_discrepancy(outputs1, outputs2)
            jsd_dipcepancy += compute_jsd_discrepancy(outputs1, outputs2)
            l2_discrepancy += compute_l2_discrepancy(outputs1, outputs2)
    classifcation_discrepancy /= num_samples
    jsd_dipcepancy /= num_samples
    l2_discrepancy /= num_samples
    return classifcation_discrepancy, jsd_dipcepancy, l2_discrepancy