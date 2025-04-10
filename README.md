# NADA: Neural Acceptance-driven Approximate Specification Mining
## Requirements
1. [spot](https://spot.lrde.epita.fr/)
2. pytorch
3. tensorboard
4. graphviz

install the graphviz with the following command.

```
apt install graphviz
```

## Start training

The following command will start training with the best hyper-parameters and will evaluate the FSM obtained from the above training process. 

```
python -u train.py --tag=0 --data_dir=data_ori_split10_neg3 --k_g=5 --K=10 --val_k=0 --lr=1e-2 --neighbor_init=1 --faithfula1=2 --faithfula2=0.8 --generate_neg=1 --sample_unlabel=1 --generate_test_neg=0 --generate_valid_neg=1 --no_valid=0 --save_best=1 --w_loss_pos=128 --num_models=1024 --gpu=1 --seed=1 --postprocess1=1 --postprocess2=1  --postprocess3=1
```

## Evaluation

The following command will evaluate the FSM obtained from the above training process:

```
python cmp_2fsa.py --tag=0 --test_case=100
```

convert the obtained FSM to graph:

```
python convert_fsa_to_dot.py --tag=0
```

## Source Files

### train.py

This is the main script for training neural network and interpreting FSM from neural network.

The arguments are as follows.

```--tag ``` An integer used as ID for every experiment. The output FSMs of experiment with ```--tag=0``` will be saved in ```model/0/```.

```--data_dir``` The path to the dataset.

```--samples``` An integer determines the number of epoch in the training process. With ```--samples=n``` and a dataset with $m$ traces , the number of epoch will be $\frac{n}{m}$.

```--k_g``` It is $N_s^m$  described in the paper.

```--batch_size``` Batch size for training neural network.

```--lr``` Learning rate for training neural network.

```--K``` Number of pieces of data. We divide the data into 10 pieces, so set it to 10.

```--val_k``` The pieces of data to be used as test set.

```--faithfula1``` It is $\beta$ described in the paper.

```--faithfula2``` It is $\gamma$ described in the paper.

```--generate_neg``` Set to $1$ to use possible negative traces and $0$ to use original traces only.

```--neighbor_init``` Set to $1$ to force that the initial states can only transfer to other states by initialization action.

```--postprocess2``` post-process, Set to $1$ to force that the initial states can only transfer to other states.

```--postprocess2``` post-process, Set to $1$ to remove unreachable states.

```--postprocess3``` post-process, Set to $1$ to remove those states not leading to accepting states.

### cmp_2fsa.py

This is the main script for evaluating FSM.

The arguments are as follows.

```--tag``` An integer used as ID for every experiment. 

```--data_dir``` The path to the dataset.

```--test_case``` Number of samples for evaluating.

```--val_k``` The pieces of data to be used as test set.

## Datasets

The folder ```data_ori_split10_neg3``` contains all data used in our evaluation.

```data_ori_split10_neg3/*/gt_fsm.txt``` are the ground truth FSMs.

```data_ori_split10_neg3/*/input.txt``` are the original traces.

```data_ori_split10_neg3/*/input.json``` are the original traces in JSON format that our model can use as input. They are divided into ```data_ori_split10_neg3/*/input_[0-9].json``` . 

```data_ori_split10_neg3/*/input_train_valk[0-9].json``` are the union of all traces except valk.

## Citation

Please consider citing the following paper if you find our codes helpful. Thank you!
