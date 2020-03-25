from pycocotools.coco import COCO as COCO_
from pycocotools.coco import _isArrayLike
import pycocotools.mask as mask_utils
import itertools
from image_loader import ImageLoader
from vis_utils import vis_one_image_opencv
import cv2
import numpy as np

def mask2poly(binarymask):
    """
    Convert a numpy binary mask to a coco-format polygon.
    :param  binarymask (numpy array)          : numpy binary mask  
    :return poly (list of list, [[], []])     : polygon of single object in coco format.
    """
    contours, _ = cv2.findContours(binarymask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    poly = [np.squeeze(arr).reshape(1,-1)[0].tolist() for arr in contours if len(arr) > 2]
    return poly

def poly2rle(poly, height, width):
    """
    Convert a coco polygon to a rle mask based on pycocotools
    :param  poly   (list of list, [[], []])   : polygon of single object in coco format.
            height (int)                      : image height
            width  (int)                      : image width
    :return rle    (rle dict)                 : rle format mask
    """
    poly = [np.array(p) for p in poly]
    rles = mask_utils.frPyObjects(poly, height, width)
    rle = mask_utils.merge(rles)
    return rle

def poly2mask(poly, height, width):
    """
    Convert a coco polygon to a binary mask based on pycocotools
    Note. This function may case slight difference in polygon.
    Eg.   poly = mask2poly(poly2mask(poly_ori))
    Due to the reason that we implement mask2poly base on opencv, 
    there are tiny differences between poly and poly_ori. 
    :param  poly   (list of list, [[], []])   : polygon of single object in coco format.
            height (int)                      : image height
            width  (int)                      : image width
    :return mask   (uint8 numpy array)        : binary mask
    """ 
    rle = poly2rle(poly, height, width)
    mask = mask_utils.decode(rle)
    return mask

def cvpoly2mask(poly, height, width):
    """
    Convert a coco polygon to a numpy binary mask based on opencv. 
    This function is slightly faster than `poly2mask`
    :param  poly   (list of list, [[], []])   : polygon of single object in coco format.
            height (int)                      : image height
            width  (int)                      : image width
    :return mask   (uint8 numpy array)        : binary mask
    """
    mask = np.zeros((height,width), dtype=np.int8)
    f = lambda x: np.array([x]).reshape((-1,2)).astype(np.int)
    ss = [f(s) for s in poly]
    mask = cv2.fillPoly(mask, ss, 255)
    mask = (mask > 20).astype(np.uint8)
    return mask

class COCO(COCO_):
    """
    Extensions for pycocotools.cooo.COCO. providing handy functions to show cocodatasets.
    """
    def __init__(self, annotation_file):
        COCO_.__init__(self, annotation_file)
        self.imloader = ImageLoader('opencv', 'bgr')
        self.annotation_file = annotation_file
        self.showInfo()

    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None, typeIds=[]):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
               catIds  (int array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)
               typeIds (int array)     : get anns for given type ids
        :return: ids (int array)       : integer array of ann ids
        """
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]
        typeIds = typeIds if _isArrayLike(typeIds) else [typeIds]

        if len(imgIds) == len(catIds) == len(areaRng) == len(typeIds) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(imgIds) == 0:
                lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset['annotations']
            anns = anns if len(catIds)  == 0 else [ann for ann in anns if ann['category_id'] in catIds]
            anns = anns if len(areaRng) == 0 else [ann for ann in anns if ann['area'] > areaRng[0] and ann['area'] < areaRng[1]]
            anns = anns if len(typeIds) == 0 else [ann for ann in anns if ann['type_id'] in typeIds]
        if not iscrowd == None:
            ids = [ann['id'] for ann in anns if ann['iscrowd'] == iscrowd]
        else:
            ids = [ann['id'] for ann in anns]
        return ids 

    def showInfo(self):
        print('* Json file: {}'.format(self.annotation_file))
        info = {
                 'Image Number': len(self.imgs),
                 'Anno Number': len(self.anns),
                 'Categories': list(self.catToImgs.keys()),
               }
        for k, v in info.items():
            print('  {:<20s} {}'.format(k,v))


    def showImgs(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None, typeIds=[]):
        """
        Show images and annotations that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
               catIds  (int array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        # remove imput image ids which not in annotation file
        filter_imgIds = []
        for i in imgIds:
            if i in self.imgs.keys():
                filter_imgIds.append(i)
        img_ids = self.getImgIds(filter_imgIds, catIds)

        for imid in img_ids:
            image = self.loadImgs(imid)[0]
            anids = self.getAnnIds(imid, catIds, areaRng, iscrowd, typeIds=typeIds)
            if len(anids) == 0: continue
            annos = self.loadAnns(anids)
            
            print(image['file_name'])
            image_arr = self.imloader.load(image['file_name'])
            boxes = [a['bbox'] for a in annos]
            if len(boxes) == 0: continue
            segms = [a['segmentation'] for a in annos]
            tags = ['type: {}, cat: {}, iscrowd: {}'.format(a['type_id'], a['category_id'],a['iscrowd']) for a in annos]
            image_show = vis_one_image_opencv(image_arr,boxes,segms,tags,auto_color=True)
            # add id tag
            cv2.putText(image_show, 'imid: {}'.format(imid), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), lineType=cv2.LINE_AA)
            #wname = 'imid: {}'.format(imid)
            wname = 'image'
            cv2.imshow(wname, image_show)
            cv2.waitKey()
            #cv2.destroyWindow(wname)

if __name__=='__main__':
    import sys
    #cc = COCO(sys.argv[1])
    #typeIds = [int(sys.argv[2])] if len(sys.argv) > 2 else []
    #imids = list(map(int, sys.argv[2].split(','))) if len(sys.argv) > 2 else []
    #cc.showImgs(imgIds=imids, typeIds=[]) 

    import time
    poly = [[1175, 416, 1172, 417, 1169, 419, 1166, 420, 1160, 420, 1155, 421, 1154, 422, 1151, 422, 1147, 421, 1145, 420, 1141, 417, 1135, 417, 1134, 416, 1130, 416, 1131, 418, 1134, 418, 1136, 419, 1141, 423, 1146, 427, 1148, 429, 1150, 432, 1151, 434, 1152, 439, 1155, 444, 1156, 450, 1156, 460, 1159, 467, 1160, 474, 1160, 478, 1159, 483, 1157, 488, 1156, 490, 1152, 494, 1151, 497, 1149, 499, 1148, 499, 1145, 502, 1143, 503, 1141, 504, 1132, 507, 1128, 508, 1119, 509, 1115, 511, 1111, 511, 1109, 512, 1107, 513, 1104, 515, 1100, 519, 1099, 521, 1103, 523, 1125, 523, 1130, 522, 1137, 519, 1139, 518, 1141, 516, 1145, 515, 1150, 511, 1160, 500, 1161, 499, 1163, 497, 1168, 492, 1170, 491, 1174, 490, 1178, 490, 1181, 488, 1185, 487, 1195, 487, 1198, 486, 1200, 485, 1206, 479, 1207, 474, 1207, 450, 1206, 444, 1203, 439, 1202, 436, 1202, 432, 1201, 430, 1199, 428, 1198, 424, 1196, 421, 1192, 417, 1190, 416, 1184, 415], [1096, 413, 1094, 414, 1092, 415, 1088, 418, 1090, 419, 1091, 420, 1096, 420, 1104, 419, 1109, 418, 1119, 414, 1127, 414, 1123, 414, 1122, 413]]
    h = 1080
    w = 1362

    st = time.time()
    for i in range(100):
        mask = cvpoly2mask(poly, h, w)
        poly_t = mask2poly(mask)
        print(poly_t)
        #print(mask)
        cv2.imshow('mask', mask*255)
        cv2.waitKey()
    print((time.time()-st) / 100)

   
