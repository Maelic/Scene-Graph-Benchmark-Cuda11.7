import argparse

def default_argument_parser(epilog="PyTorch Relation Detection Training"):
    """
    Create a parser with some common arguments.
    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.
    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(epilog=epilog)

    parser.add_argument("--config-file",
        type=str,
        default="",
        metavar="FILE",
        help="path to config file",
        required=True
    )
    parser.add_argument("--local_rank", type=int, default=0)

    parser.add_argument("--skip-test",
        type=bool,
        dest="skip_test",
        help="Do not test the final model",
        action="store_true"
    )
    parser.add_argument("--use-wandb",
        type=bool,
        dest="use_wandb",
        help="Use wandb logger (Requires wandb installed)",
        action="store_true",
        default=False
    )

    parser.add_argument("--verbose",
        type=bool,
        dest="verbose",
        help="Print more information",
        action="store_true",
        default=False
    )

    parser.add_argument("--task",
        type=str,
        dest="task",
        help="Chose between precls, sgcls or sgdet",
        default="precls",
        required=True
    )

    parser.add_argument("--save_best",
        type=bool,
        dest="save_best",
        action="store_true",
        help="Only save the best epoch to save space",
    )

    parser.add_argument("opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    return parser
