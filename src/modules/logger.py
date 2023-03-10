import logging
import os

logging.root.setLevel(logging.NOTSET)


def init_logger(args, name):
    if name:
        logger = logging.getLogger(name)
    else:
        name = os.path.basename(args.log_path)
        logger = logging.getLogger(name)

    if not logger.handlers:
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(args.log_path)
        c_handler.setLevel(logging.WARNING)
        f_handler.setLevel(logging.INFO)

        c_format = logging.Formatter(
            '%(name)s - %(levelname)s - %(message)s')
        f_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)

        logger.addHandler(c_handler)
        logger.addHandler(f_handler)

    return logger


def get_global_logger(args, name=None):
    return init_logger(args, name)
