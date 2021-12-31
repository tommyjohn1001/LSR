from all_packages import *
from utils import *

dotenv.load_dotenv(override=True)

if __name__ == "__main__":
    args = get_args()

    hparams = {
        "batch_size": args.batch_size,
        "num_workers": 10,
        "lr": args.lr,
        "lr_decay": args.lr_decay,
        "decay_epoch": args.decay_epoch,
        "max_epochs": args.num_epoch,
        "gradient_clip_val": args.max_grad_norm,
        "weight_decay": 0,
        "input_theta": args.input_theta,
        "optim": args.optim,
    }

    train_loader, val_loader = get_loaders(args)
    litmodel = get_model(args, hparams)
    trainer = get_trainer(args, hparams)

    ## Log hyperparameters
    log_hparams(trainer, hparams)

    ## Start training
    trainer.fit(
        litmodel, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.ckpt
    )

    ## TODO: Consider if this is needed
    ## Start testing
    # final_test_model(path_dir_ckpt, litmodel, trainer, val_loader)
