import torch
import os
 
trial=1
num_elements=200
model = 'RatSPN'
leaf_type = 'Categorical'
leaf_config=dict(num_bins=784)
constrained = False
dataset_name = f"set-fmnist-{num_elements}"

experiment_dir = f"../experiments/{dataset_name}/{model}/leaf={leaf_type}/constrained={constrained}"
if(os.path.exists(experiment_dir)):
    trial = len(os.listdir(experiment_dir))+1
experiment_dir = os.path.join(experiment_dir, f'trial={trial}')  

config = dict(
    experiment_dir=experiment_dir,
    seed=trial,
    dataset=dict(
        name=f"set-fmnist-{num_elements}",
        datadir=f"../data/FashionMNIST/num_elements={num_elements}/",
    ),
    dataloader=dict(
        shuffle=True,
        batch_size=100,
        pin_memory=True,
    ),
    model=dict(
        name=model,
        S=10,
        I=10,
        D=6,
        R=10,
        F=num_elements,
        C=1,
        # name="RatSPN",
        # num_sums=20,
        # num_input_distributions=20,
        # num_repetition=20,
        # depth=5,
        # num_vars=50,
        # num_dims=2,
        # num_classes=1,
        # graph_type='random_binary_trees',
        leaf_type=leaf_type,
        leaf_config=leaf_config
    ),
    constraint_args=dict(
        constrained=constrained,
        type="generalization",
        atol=1e-1,
        lmbda=1,
    )
)

config["train_args"] = dict(
        num_epochs=200,
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        alpha=0.25,
        print_every=1,
        visualize_every=5,
        lr=0.1,
        results_dir=f'{experiment_dir}/results',
        print_args=["trn_loss", "val_loss", "val_acc", "tst_loss", "tst_acc", "time"],
        return_args=[],
        plots_dir=f'{experiment_dir}/plots',
        visualize=True,
        save_model_dir=f'{experiment_dir}/ckpt'
)

