import logging as l
from fastcore.all import ifnone
from pathlib import Path
from datetime import date
from collections import deque

DEF_FMT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

class LoggingQueue(deque):
    '''deque with `logging.Handler` api methods'''
    def put_nowait(self, rec): self.append(rec.message)

def init_logger(name: str = None, level=l.INFO, format: str = None, handlers: list = None, logs_dir='./logs'):
    '''Initializes a logger, adds handlers and sets the format. If logs_dir is provided, a file handler is added to the logger.'''
    handlers = ifnone(handlers, [])
    handlers.append(l.StreamHandler())
    if logs_dir: 
        p = Path(logs_dir)/f'{date.today()}.log'
        p.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(l.FileHandler(p)) 
    log_fmt = l.Formatter(ifnone(format, DEF_FMT), datefmt='%Y-%m-%d %H:%M:%S')
    log = l.getLogger(name)
    log.setLevel(level)
    log.handlers.clear()
    for h in handlers: h.setFormatter(log_fmt); log.addHandler(h)