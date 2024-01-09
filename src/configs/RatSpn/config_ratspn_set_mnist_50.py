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
        name="RatSPN",
        S=10,
        I=5,
        D=4,
        R=20,
        F=100,
        C=1,
    ),
    constraint_args=dict(
        constrained=False,
        type="generalization",
        atol=1e-1,
        lmbda=1,
    ),
    train_args=dict(
        num_epochs=300,
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        alpha=0.25,
        print_every=1,
        visualize_every=10,
        lr=0.01,
        results_dir='results/',
        print_args=["trn_loss", "val_loss", "val_acc", "tst_loss", "tst_acc", "time"],
        return_args=[],
        plots_dir="../plots/image/set-mnist-50/RatSPN",
        visualize=True,
        save_model_dir="../ckpt/image/set-mnist-50/RatSPN"
        )
)
