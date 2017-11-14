import os
import logging
from logging import handlers

# extend logging levels to be accessible here
DEBUG    = logging.DEBUG
INFO     = logging.INFO
WARNING  = logging.WARNING
ERROR    = logging.ERROR
CRITICAL = logging.CRITICAL

# capture warnings and emit logging.warning() message
logging.captureWarnings(True)

def RotatingFile(fname, name=None, level=logging.INFO, backupCount=10):
    """initialize and return standard logger and error logger objects

    Args:
        logname (str): basename for the logfile that is initialized. Error log will have name:
            <logname>_Errors.log by default
        name (str): name of logger
        level (int): log level - less severe messages will be passed to parent logger or ignored
        backupCount (int): number of log files to keep in rotation
    """
    basename = os.path.splitext(os.path.basename(fname))[0]
    dirname  = os.path.dirname(fname)
    fname = os.path.join(dirname, basename + '.log') #.format(time.strftime('%Y%b%d_%H:%M:%S')))
    # get a named logger - if not named, root will get messages from children loggers
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # create common formatter
    #  formatter = logging.Formatter(fmt='%(message)s')

    # create separate handlers for stream and file
    sh = logging.StreamHandler()  # defaults to sys.stderr
    #  sh.setFormatter(formatter)
    logger.addHandler(sh)

    rfh = logging.handlers.RotatingFileHandler(fname, mode='w', maxBytes=0, backupCount=backupCount, delay=True)
    #  rfh.setFormatter(formatter)
    logger.addHandler(rfh)

    # error logger
    fname_err = os.path.join(dirname, basename+'_errors.log')
    err_rfh = logging.handlers.RotatingFileHandler(fname_err, mode='w', maxBytes=0, backupCount=backupCount, delay=True)
    #  err_rfh.setFormatter(formatter)
    err_rfh.setLevel(logging.WARNING)
    logger.addHandler(err_rfh)

    # Initialize Loggers
    os.makedirs(dirname, exist_ok=True)
    rfh.doRollover()
    err_rfh.doRollover()

    # return loggers
    return logger
