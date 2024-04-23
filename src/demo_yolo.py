from functools import partial
import cv2
import fire
import numpy as np
from ultralytics import YOLO
import logging as l
from utils import init_logger

def extract_points(mask): return mask.xy[0].astype(np.int32)[None]


def get_mask(result):
    '''get segmentation mask from yolo model'''
    res = np.zeros(result.orig_shape, dtype=np.uint8)
    for mask in result.masks:
        res = cv2.fillPoly(res, extract_points(mask), 255)
    return res

def merge_with_mask(image, mask, p=0.2, gamma=0):
    '''Merge together original image and segmentation mask'''
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)*np.array([0,1,0], np.uint8)
    return cv2.addWeighted(image, 1-p, mask_color, p, gamma)


def poly_area_np(points):
    '''get area of polygon using Shoelace formula'''
    if not np.array_equal(points[0], points[-1]):
        points = np.concatenate([points, [points[0]]])
    x,y = points.T
    return 0.5 * np.abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) 
                        + (x[-1] * y[0] - x[0] * y[-1]))


def add_area_labels(img, result):
    '''add labels on image based on yolo output'''
    image = img.copy()
    for m in result.masks:
        points = extract_points(m)[0]
        text = str(poly_area_np(points)/1000)
        x,y = points.mean(0).astype(np.int32)
        image = cv2.putText(image, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX ,  
                   1, (255, 0, 0), 2, cv2.LINE_AA) 
    return image


def process_and_display(file, func=None):
    if func is None:  func = lambda x: x

    video = cv2.VideoCapture(file)
    l.info(f'captured video from {file}')

    try:
        while True:
            ret, frame = video.read()
            if not ret: break
            l.info('read frame')
            out = func(frame)
            l.info('processed frame')
            cv2.imshow('frame', out)
            if cv2.waitKey(1) == ord('q'): break
    except KeyboardInterrupt:
        l.warn('Get keyboard interrupt')
    finally:
        video.release()
        cv2.destroyAllWindows()

        
def segment_frame(frame, yolo): 
    res = yolo(frame, verbose=False)[0]
    l.info('model got predictions')
    out = add_area_labels(merge_with_mask(frame,get_mask(res)), res)
    l.info('postprocessing done')
    return out

### cli ###
def demo(video: str, model_path: str='./models/yolo.pt'):
    yolo = YOLO(model_path, 'segment')
    l.info(f'model loaded from {model_path}')
    func = partial(segment_frame, yolo=yolo)
    process_and_display(video, func)

if __name__=='__main__':
    init_logger(logs_dir=None)
    l.info('started demo')
    fire.Fire(demo)