import torch

num_elements=100
config = dict(
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
        name="EinsumNet",
        num_sums=10,
        num_input_distributions=10,
        num_repetition=5,
        depth=6,
        num_vars=num_elements,
        num_dims=1,
        num_classes=1,
        graph_type='random_binary_trees',
        leaf_type='CategoricalArray',
        leaf_config=dict(K=784)
        # leaf_type='NormalArray',
        # leaf_config=dict()
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
        visualize_every=5,
        lr=0.1,
        results_dir='results/',
        print_args=["trn_loss", "val_loss", "val_acc", "tst_loss", "tst_acc", "time"],
        return_args=[],
        plots_dir=f'../plots/set-mnist/num_elements={num_elements}/{config["dataset"]["name"]}/{config["model"]["name"]}/leaf={config["model"]["leaf_type"]}/constrained={config["constraint_args"]["constrained"]}',
        visualize=True,
        save_model_dir=f'../ckpt/set-mnist/num_elements={num_elements}/{config["dataset"]["name"]}/{config["model"]["name"]}/leaf={config["model"]["leaf_type"]}/constrained={config["constraint_args"]["constrained"]}'
)
