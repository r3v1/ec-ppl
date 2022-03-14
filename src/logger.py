import logging
from pathlib import Path


def make_logger(name: str, working_dir: Path = None):
    # Logging
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # File handler
    logs_dir = working_dir or Path("~/.evolution").expanduser()
    if not logs_dir.is_dir():
        logs_dir.mkdir(parents=True, exist_ok=True)
    
    fh = logging.FileHandler(logs_dir / f"{name.lower()}.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger
