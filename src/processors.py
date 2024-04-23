import torch, numpy as np, cv2, math
from torchvision import transforms as TF
from fastcore.all import Pipeline, Transform
from fastai.vision.all import ToTensor, IntToFloatTensor, PILImageBW, TensorImage, TensorImageBW


@Transform
def gamma_tfm(x: TensorImage|TensorImageBW, gamma=0.8):
    return x**gamma

class ImageProcessor:
    '''Encodes image into tensor, decodes back into image and applies different image effects'''
    def __init__(self, gamma=0.8):
        self.gamma = gamma
        self.tfms = Pipeline([ToTensor(), IntToFloatTensor(), TF.Resize((256, 256))])
    
    def encode(self, image):
        res = self.tfms(PILImageBW.create(image))[0]
        return gamma_tfm(res, gamma=self.gamma)

    def decode(self, tensor):
        tensor = (tensor*255).numpy().astype(np.uint8)
        return cv2.cvtColor(tensor, cv2.COLOR_GRAY2BGR)
    
    def add_texts(self, image, texts, coords, size=1, color=(255,0,0), thickness=1):
        '''Place texts at coordinets'''
        image = image.copy()
        for i, t in enumerate(texts):
            image = cv2.putText(image, str(t), tuple(coords[i]), cv2.FONT_HERSHEY_SIMPLEX,
                                size, color, thickness, cv2.LINE_AA) 
        return image

    def merge_with_mask(self, image, mask, p=0.2, gamma=0):
        '''Merge together original image and segmentation mask'''
        mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)*np.array([0,1,0], np.uint8)
        return cv2.addWeighted(image, 1-p, mask_color, p, gamma)

class MaskProcessor:
    '''Applies postprocessing to output tensor to get final mask and stats'''
    def _init_(self): pass
    
    def decode_mask(self, pred, ths=0.5):
        mask = torch.sigmoid(pred[0]).cpu()[0]
        return IntToFloatTensor().decodes(mask>ths).numpy().astype(np.uint8)
    
    def detach_blobs(self, mask, scale=2):
        kernel = np.ones((scale, scale), np.uint8)
        mask = cv2.erode(mask,kernel,iterations = 1)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    def get_components(self, mask, area_ths = 0.1):
        _, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        #cast to normal types
        stats = stats.astype(np.float32)
        centroids = centroids.astype(np.int32)
        labels = labels.astype(np.uint8)
        # get normalized resolution-independent areas multiplied by 100 for convenience
        stats[:,cv2.CC_STAT_AREA] /= math.prod(mask.shape)/100
        mask = (stats[:,cv2.CC_STAT_AREA]>area_ths).nonzero()
        labels[~np.isin(labels, mask)]=0
        labels[labels!=0]=255
        # [1:] to not take the background (thank you opencv...)
        return labels, stats[mask][1:], centroids[mask][1:]