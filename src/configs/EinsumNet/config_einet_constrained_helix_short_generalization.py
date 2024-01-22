import torch
import os

trial = 1
model = 'EinsumNet'
leaf_type = 'NormalArray'
leaf_config=dict(K=3)
constrained = True
dataset_name = "helix_short_appended"
experiment_dir = f"../experiments/{dataset_name}/{model}/leaf={leaf_type}/constrained={constrained}"

if(os.path.exists(experiment_dir)):
    trial = len(os.listdir(experiment_dir))+1
experiment_dir = os.path.join(experiment_dir, f'trial={trial}')
config = dict(
    experiment_dir=experiment_dir,
    seed=trial,
    dataset=dict(
        name=dataset_name,
        datadir="../data/toy_3d",
    ),
    dataloader=dict(
        shuffle=True,
        batch_size=100,
        pin_memory=True,
    ),
    model=dict(
        name="EinsumNet",
        num_sums=10,
        num_input_distributions=10,
        num_repetition=10,
        depth=1,
        num_vars=3,
        num_dims=1,
        num_classes=1,
        graph_type='random_binary_trees',
        leaf_type=leaf_type,
        leaf_config=leaf_config
    ),
    constraint_args=dict(
        constrained=constrained,
        type="generalization",
        atol=1e-1,
        lmbda=10,
    )
)

config["train_args"] = dict(
        num_epochs=200,
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        alpha=0.25,
        print_every=1,
        visualize_every=10,
        lr=0.01,
        results_dir=f'{experiment_dir}/results',
        print_args=["trn_loss", "val_loss", "val_acc", "tst_loss", "tst_acc", "time"],
        return_args=[],
        plots_dir=f'{experiment_dir}/plots',
        visualize=True,
        save_model_dir=f'{experiment_dir}/ckpt'
)
