import logging as l, cv2
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

 
def process_and_display(file, func=None):
    if func is None:  func = lambda x: x
    video = cv2.VideoCapture(file)
    l.info(f'captured video from {file}')
    try:
        while True:
            ret, frame = video.read()
            if not ret: break
            l.info('read frame')
            out, stats = func(frame)
            if stats is not None: l.info(f'processed frame, total clods: {len(stats)} | largest area: {stats.max()}')
            cv2.imshow('frame', cv2.resize(out, (960, 540)))
            key = cv2.waitKey(1)
            if key == ord(' '): cv2.waitKey(-1)
            if key == 27: break
    except KeyboardInterrupt:
        print('Get keyboard interrupt')
    finally:
        video.release()
        cv2.destroyAllWindows()
