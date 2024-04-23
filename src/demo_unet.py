import cv2, torch, numpy as np, logging as l
from fastai.vision.all import *
from unet import UNet2DModel
from processors import ImageProcessor, MaskProcessor
from utils import init_logger, process_and_display
import fire

def read_yolo(file):
    res = []
    with open(file, 'r') as f:
        for l in f: res.append(np.array(l.split()[1:], dtype=np.float32).reshape(-1,2))
    return res

def create_mask(img_shape, labels):
    res = np.zeros(img_shape, dtype=np.uint8)
    h,w = img_shape
    for mask in labels: res = cv2.fillPoly(res, [(mask*(w,h)).astype(np.int32)], 255)
    return res

def segment_frame(frame, model):
    ## MAIN CONFIGURATION
    area_ths, blob_scale, prob_ths, gamma, mask_prop = AREA_THS, BLOB_SCALE, PROB_THS, GAMMA, MASK_PROP
    # encode and prepare
    proc = ImageProcessor()
    inp = proc.encode(frame)
    out = proc.decode(inp)
    # predict
    res = model(to_device(inp[None,None]))
    # prepare mask
    m_proc = MaskProcessor()
    mask = m_proc.decode_mask(res, prob_ths)
    mask = m_proc.detach_blobs(mask, blob_scale)
    labels, stats, centroids = m_proc.get_components(mask, area_ths)
    # merge and add texts
    out = proc.merge_with_mask(out, labels, mask_prop, gamma)
    out = proc.add_texts(out, stats[:,cv2.CC_STAT_AREA].round(2), centroids, 0.3)
    texts = [f'total: {len(stats)}', f'biggest: {str(round(stats[:,cv2.CC_STAT_AREA].max(),2))}']
    out = proc.add_texts(out, texts, [(2,10), (2, 25)], 0.4)
    return out, stats[:,cv2.CC_STAT_AREA]



############ CLI ############

def demo(video: str, model_path: str = './models/unet.pt'):
    '''launch demo'''
    model = torch.load(model_path, map_location=default_device())
    process_and_display(video, partial(segment_frame, model=model))


def train(data_path: str, save_path: str = './models/unet.pt', n_epoch=10):
    '''train u-net model'''
    @ItemTransform
    def mask_tfm(item):
        x, y = item
        y = create_mask(x.shape, y).astype(np.float32)
        return x, TensorMask(y/255.)

    @Transform
    def gamma_tfm(x: TensorImage|TensorImageBW, gamma=0.8): return x**gamma
    
    dblock = DataBlock(blocks=(ImageBlock(PILImageBW), TransformBlock(batch_tfms=IntToFloatTensor())),
                   get_items=lambda p: get_image_files(p, folders=['valid','train']),
                   get_y=lambda o: read_yolo(str(o).replace('images', 'labels').replace('jpg','txt')),
                   splitter=GrandparentSplitter(valid_name='valid'),
                   item_tfms=[mask_tfm],
                   batch_tfms=[gamma_tfm, *aug_transforms(size=256, min_scale=0.75, max_lighting=0.2)])
    
    dls = dblock.dataloaders(data_path, bs=4)
    l.info('created dataset')
    model = UNet2DModel()
    learn = Learner(dls, model, loss_func=BCEWithLogitsLossFlat(), metrics=[])
    l.info(f'start training for {n_epoch} epochs')
    learn.fit(n_epoch, cbs = [MixedPrecision()])
    torch.save(model, save_path)
    

if __name__=='__main__':
    ## CONFIG PARAMETERS, basically defaults are OK, but you can play around
    AREA_THS, BLOB_SCALE, PROB_THS, GAMMA, MASK_PROP = 0.1, 2, 0.5, 20, 0.5
    init_logger()
    fire.Fire({'train': train, 'demo': demo})
