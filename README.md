How to run:

To run a single resnet-discrepncy expirement, and replicate our results, use train_resnet.py like so:

train_resnet.py --epoch=150 --model_name=resnet18  --noise_level=0.5

Addiotnaly parameters (such as optimizer,weight_decay,..) can be found in "train_resnet.py" or use --help

To plot multiple resnet's at the same graph, use "train_resnet2.py"

train_resnet2.py --noise_level=0.3 --dataset=cifar100 --lr=0.01 --weight_decay=5e-4 --momentum=0.9 --epochs=200 --batch_size=128

To intiate both networks with very small epsilon between them, "train_resnet3.py"
