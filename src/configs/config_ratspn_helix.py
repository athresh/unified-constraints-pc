import torch
config = dict(
    dataset=dict(
        name="helix",
        datadir="../data/toy_3d",
    ),
    dataloader=dict(
        shuffle=True,
        batch_size=200,
        pin_memory=True,
    ),
    model=dict(
        name="RatSPN",
        S=10,
        I=3,
        D=1,
        R=20,
        F=3,
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
        plots_dir="../plots/toy_3d/helix/RatSPN",
        visualize=True,
        save_model_dir="../ckpt/toy_3d/helix/RatSPN"
        )
)
