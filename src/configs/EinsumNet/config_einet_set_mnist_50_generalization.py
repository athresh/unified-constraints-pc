import torch
config = dict(
    dataset=dict(
        name="set-mnist-50",
        datadir="../data/MNIST",
    ),
    dataloader=dict(
        shuffle=True,
        batch_size=200,
        pin_memory=True,
    ),
    model=dict(
        name="EinsumNet",
        num_sums=10,
        num_input_distributions=10,
        num_repetition=10,
        depth=4,
        num_vars=50,
        num_dims=2,
        num_classes=1,
        graph_type='random_binary_trees',
        leaf_config=dict()
    ),
    constraint_args=dict(
        constrained=True,
        type="generalization",
        atol=1e-1,
        lmbda=1,
    )
)

config["train_args"] = dict(
        num_epochs=500,
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        alpha=0.25,
        print_every=1,
        visualize_every=10,
        lr=0.01,
        results_dir='results/',
        print_args=["trn_loss", "val_loss", "val_acc", "tst_loss", "tst_acc", "time"],
        return_args=[],
        plots_dir=f'../plots/set-mnist/{config["dataset"]["name"]}/{config["model"]["name"]}',
        visualize=True,
        save_model_dir=f'../ckpt/set-mnist/{config["dataset"]["name"]}/{config["model"]["name"]}'
)
