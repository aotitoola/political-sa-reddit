import logging
import time
import sys

# set loggers
loggers = {}


def set_logger(name, level):
    global loggers

    if loggers.get(name):
        return loggers.get(name)
    else:
        logger = logging.getLogger(name)
        if logger.hasHandlers():
            logger.handlers.clear()

        logger.setLevel(level)

        _ = time.strftime("%Y.%m.%d_%H.%M.%S", time.localtime())
        formatter = logging.Formatter('[%(asctime)s][%(levelname)s] ## %(message)s')

        fh = logging.FileHandler('./logs/' + name + '.log')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        loggers[name] = logger
        return logger


logger = set_logger('reddit_sa', logging.DEBUG)
