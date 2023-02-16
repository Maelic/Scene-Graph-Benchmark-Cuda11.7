# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os
import sys

DEBUG_PRINT_ON = True

def debug_print(logger, info):
    if DEBUG_PRINT_ON:
        logger.info('#'*20+' '+info+' '+'#'*20)

def setup_logger(name, save_dir, distributed_rank, filename="log.txt"):
    try:
        from loguru import logger

        logger.add(sys.stdout, format="{time} {level} {message}", filter="my_module", level="DEBUG")
        logger.info("Using loguru logger")
        if distributed_rank > 0:
            return logger
        if save_dir:
            logger.add(os.path.join(save_dir, filename))

    except ImportError:
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.info("Using default logger")
        # don't log results for the non-master process
        if distributed_rank > 0:
            return logger
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if save_dir:
            fh = logging.FileHandler(os.path.join(save_dir, filename))
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger
