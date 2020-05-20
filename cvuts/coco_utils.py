from pycocotools.coco import COCO as COCO_
from pycocotools.coco import _isArrayLike
import pycocotools.mask as mask_utils
import itertools
from cvuts.image_loader import ImageLoader
from cvuts.vis_utils import vis_one_image_opencv
import cv2
import numpy as np

def mask2poly(binarymask):
    """
    Convert a numpy binary mask to a coco-format polygon.
    :param  binarymask (numpy array)          : numpy binary mask  
    :return poly (list of list, [[], ...])    : polygon of single object in coco format.
    """
    contours, _ = cv2.findContours(binarymask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    poly = [np.squeeze(arr).reshape(1,-1)[0].tolist() for arr in contours if len(arr) > 2]
    return poly

def poly2rle(poly, height, width):
    """
    Convert a coco polygon to a rle mask based on pycocotools
    :param  poly   (list of list, [[], ...])  : polygon of single object in coco format.
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
    :param  poly   (list of list, [[], ...])  : polygon of single object in coco format.
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
    :return mask   (unit8 numpy array)              : binary mask
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
                 'Categories': self.dataset['categories'], 
               }
        for k, v in info.items():
            print('  {:<20s} {}'.format(k,v))

        cats = list(self.catToImgs.keys())
        cats.sort()
        cnts = []
        names = []
        print('* Exist cats')
        for c in cats:
            cnts.append(len(self.getAnnIds(catIds=[c])))
            names.append(self.cats[c]['name'])
        for c, n, cnt in zip(cats, names, cnts):
            print(' '*2+'{:<5s} {:<30s} {:<10s}'.format(str(c),str(n),str(cnt)))

    def showImgs(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None, typeIds=[], skip=1):
        """
        Show images and annotations that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
               catIds  (int array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        # remove imput image ids which not in annotation file
        filter_imgIds = []  # make sure that [] would not pass to getImgIds if no imid in dataset
        for i in imgIds:
            if i in self.imgs.keys():
                filter_imgIds.append(i)
        if len(filter_imgIds) > 0 or len(imgIds)==0:
            img_ids = self.getImgIds(filter_imgIds, catIds)
        else:
            img_ids = []
        #print(img_ids)

        img_cnt = 0
        for idx, imid in enumerate(img_ids):
            if (idx+1) % skip != 0:
                continue
            image = self.loadImgs(imid)[0]
            anids = self.getAnnIds(imid, catIds, areaRng, iscrowd, typeIds=typeIds)
            if len(anids) == 0: continue
            img_cnt += 1
            annos = self.loadAnns(anids)
            
            print(image['file_name'])
            image_arr = self.imloader.load(image['file_name'])
            boxes = [a['bbox'] for a in annos]
            if len(boxes) == 0: continue
            print(annos[0])
            if annos[0]['segmentation']:
                segms = [a['segmentation'] for a in annos if len(a['segmentation']) > 0 and len(a['segmentation'][0])>2]
            else:
                segms = None

            tags = build_tag(annos, [1, len(areaRng)>0, iscrowd is not None, len(typeIds)>0])
            #tags = ['type: {}, cat: {}, iscrowd: {}'.format(a['type_id'], a['category_id'],a['iscrowd']) for a in annos]
            image_show = vis_one_image_opencv(image_arr,boxes,segms,tags,auto_color=True)
            # add id tag
            cv2.putText(image_show, 'imid: {}'.format(imid), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), lineType=cv2.LINE_AA)

            wname = 'image'
            cv2.imshow(wname, image_show)
            cv2.waitKey()
        print('total related images: {}'.format(img_cnt))


def build_tag(annos, fields=[0,0,0,0]):
    t = ['cat: {}', 'area: {}', 'iscrowd: {}', 'type: {}']
    fn = ['category_id', 'area', 'iscrowd', 'type_id']
    fm =  [t[i] for i in range(len(fields)) if fields[i] ]
    template =  ', '.join(fm)
    tags = [template.format(*(a[fn[i]] for i in range(len(fields)) if fields[i])) for a in annos]
    return tags
