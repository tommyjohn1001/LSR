from pytorch_lightning import seed_everything

from all_packages import *
from src.datasets.dataset import IterDataset
from src.models import *

seed_everything(42, workers=True)


def log_hparams(trainer, hparams):
    trainer.logger.log_hyperparams(hparams)


def get_args():
    parser = argparse.ArgumentParser()

    # configurations for data
    parser.add_argument("--data_path", type=str, default="code/prepro_data")

    parser.add_argument("--superpod", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--ckpt", "-c", default=None)
    parser.add_argument("--gpus", "-g", default="0")

    parser.add_argument(
        "--model_name",
        type=str,
        default="LSR",
        help="[LSR, LSR_Bert], name of the model",
        choices=["LSR", "LSR_Bert"],
    )

    parser.add_argument("--train_prefix", type=str, default="dev_train")
    parser.add_argument("--test_prefix", type=str, default="dev_dev")

    # configurations for model
    parser.add_argument(
        "--finetune_emb", type=int, default=False, help="Fine tune pre-trained word Embedding."
    )
    parser.add_argument("--use_struct_att", type=bool, default=False)
    parser.add_argument("--use_reasoning_block", type=bool, default=True)
    parser.add_argument(
        "--reasoner_layer_first",
        type=int,
        default=3,
        help="Num of the second sublayers in dcgcn block.",
    )
    parser.add_argument(
        "--reasoner_layer_second",
        type=int,
        default=4,
        help="Num of the second sublayers in dcgcn block.",
    )

    # configurations for dimension
    parser.add_argument("--emb_dim", type=int, default=EMB_DIM, help="Word embedding dimension.")
    parser.add_argument(
        "--coref_dim", type=int, default=20, help="Coreference embedding dimension."
    )
    parser.add_argument("--ner_dim", type=int, default=20, help="NER embedding dimension.")
    parser.add_argument("--pos_dim", type=int, default=20, help="POS embedding dimension.")
    parser.add_argument(
        "--hidden_dim", type=int, default=HIDDEN_DIM, help="RNN hidden state size."
    )

    # configurations for dropout
    parser.add_argument("--dropout_emb", type=float, default=0.2, help="embedding dropout rate.")
    parser.add_argument("--dropout_rnn", type=float, default=0.2, help="rnn dropout rate.")
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="dropout rate.")
    parser.add_argument("--dropout_gcn", type=float, default=0.4, help="GCN dropout rate.")

    # configurations for training
    parser.add_argument("--lr", type=float, default=LR, help="Applies to sgd and adagrad.")
    parser.add_argument(
        "--lr_decay", type=float, default=DECAY_RATE, help="Learning rate decay rate."
    )
    parser.add_argument(
        "--decay_epoch", type=int, default=20, help="Decay learning rate after this epoch."
    )
    parser.add_argument(
        "--evaluate_epoch", type=int, default=30, help="Evaluate after this epoch."
    )
    parser.add_argument(
        "--optim",
        choices=["sgd", "adagrad", "adam", "adamax"],
        default="adam",
        help="Optimizer: sgd, adagrad, adam or adamax.",
    )
    parser.add_argument(
        "--num_epoch", type=int, default=MAX_EPOCH, help="Number of total training epochs."
    )
    parser.add_argument(
        "--batch_size", "-b", type=int, default=BATCH_SIZE, help="Training batch size."
    )
    parser.add_argument("--max_grad_norm", type=float, default=5.0, help="Gradient clipping.")
    parser.add_argument("--log_step", type=int, default=30, help="Print log every k steps.")
    parser.add_argument("--log", type=str, default="logs.txt", help="Write training log to file.")
    parser.add_argument(
        "--save_epoch", type=int, default=100, help="Save model checkpoints every k epochs."
    )
    parser.add_argument(
        "--save_dir", type=str, default="./saved_models", help="Root dir for saving models."
    )
    parser.add_argument(
        "--id", type=str, default="00", help="Model ID under which to save models."
    )
    parser.add_argument("--info", type=str, default="", help="Optional info for the experiment.")
    parser.add_argument("--input_theta", type=float, default=-1)

    # others
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--cuda", type=bool, default=torch.cuda.is_available())
    parser.add_argument("--cpu", action="store_true", help="Ignore CUDA.")
    parser.add_argument(
        "--save_name",
        type=str,
        default=NAME
        + "_DIM_"
        + str(EMB_DIM)
        + "_HIDDEN_"
        + str(HIDDEN_DIM)
        + "_"
        + data_set
        + "_LR_"
        + str(LR)
        + "_DECAY_"
        + str(DECAY_RATE)
        + "_BATCHSIZE_"
        + str(BATCH_SIZE)
        + "_SEED_"
        + str(SEED),
    )

    args = parser.parse_args()

    args.gpus = [int(x) for x in args.gpus.split(",")]

    return args


def get_trainer(args, hparams):
    root_logging = "logs"
    if not args.superpod:
        now = (datetime.now() + timedelta(hours=7)).strftime("%b%d_%H-%M-%S")
        name = f"{data_set}_{now}"
    else:
        now = datetime.now().strftime("%b%d_%H-%M-%S")
        name = f"{data_set}_superpod_{now}"

    path_dir_ckpt = osp.join(root_logging, "ckpts", name)

    ## TODO: Check this again
    callback_ckpt = ModelCheckpoint(
        dirpath=path_dir_ckpt,
        monitor="val_f1",
        filename="{epoch}-{val_loss:.3f}",
        mode="max",
        save_top_k=2,
        save_last=True,
    )
    callback_tqdm = TQDMProgressBar(refresh_rate=5)
    callback_lrmornitor = LearningRateMonitor(logging_interval="step")
    logger_tboard = TensorBoardLogger(osp.join(root_logging, "tboard"), name=name)
    logger_wandb = WandbLogger(name, root_logging)

    trainer = Trainer(
        gpus=args.gpus,
        max_epochs=hparams["max_epochs"],
        strategy="ddp" if len(args.gpus) > 1 else None,
        callbacks=[callback_ckpt, callback_tqdm, callback_lrmornitor],
        logger=logger_wandb if args.wandb is True else logger_tboard,
        # log_every_n_steps=5,
    )

    return trainer


def get_model(args, hparams):
    if args.model_name == "LSR_Bert":
        model = LSR_Bert(args)
    else:
        model = LSR(args)

    litmodel = LSRLitModel(model, hparams)

    return litmodel


def get_loaders(args):
    dataset_train = IterDataset(args, TRAIN_PREFIX)
    dataset_test = IterDataset(args, TEST_PREFIX)

    ## Load data
    dataset_train.load_data()
    dataset_test.load_data()

    dataloader_train = DataLoader(
        dataset_train, batch_size=1, batch_sampler=None, collate_fn=lambda batch: batch
    )
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=1,
        batch_sampler=None,
        shuffle=False,
        collate_fn=lambda batch: batch,
    )

    return dataloader_train, dataloader_test
