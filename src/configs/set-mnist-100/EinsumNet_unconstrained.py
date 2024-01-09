import torch
import os
 
trial = 1

num_elements=100
model = 'EinsumNet'
leaf_type = 'CategoricalArray'
leaf_config=dict(K=784)
constrained = False
dataset_name = f"set-mnist-{num_elements}"
experiment_dir = f"../experiments/{dataset_name}/{model}/leaf={leaf_type}/constrained={constrained}"

if(os.path.exists(experiment_dir)):
    trial = len(os.listdir(experiment_dir))+1
experiment_dir = os.path.join(experiment_dir, f'trial={trial}')  
config = dict(
    experiment_dir=experiment_dir,
    dataset=dict(
        name=f"set-mnist-{num_elements}",
        datadir=f"../data/MNIST/num_elements={num_elements}/",
    ),
    dataloader=dict(
        shuffle=True,
        batch_size=100,
        pin_memory=True,
    ),
    model=dict(
        name=model,
        num_sums=20,
        num_input_distributions=20,
        depth=6,
        num_repetition=5,
        num_vars=num_elements,
        num_dims=1,
        num_classes=1,
        graph_type='random_binary_trees',
        leaf_type=leaf_type,
        leaf_config=leaf_config
        # leaf_type='NormalArray',
        # leaf_config=dict()
    ),
    constraint_args=dict(
        constrained=constrained,
        type="generalization",
        atol=1e-1,
        lmbda=1,
    )
)

config["train_args"] = dict(
        num_epochs=5,
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
