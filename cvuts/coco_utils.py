from pycocotools.coco import COCO as COCO_
from pycocotools.coco import _isArrayLike
import itertools
from image_loader import ImageLoader
from vis_utils import vis_one_image_opencv
import cv2

class COCO(COCO_):
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
        print('*' * 50)
        print('Json file: {}'.format(self.annotation_file))
        print('*' * 50)
        info = {
                 'Image Number': len(self.imgs),
                 'Anno Number': len(self.anns),
                 'Categories': list(self.catToImgs.keys()),
               }
        for k, v in info.items():
            print('{:<20s} {}'.format(k,v))
        print('*' * 50)


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
            
            wname = 'imid: {}'.format(imid)
            cv2.imshow(wname, image_show)
            cv2.waitKey()
            cv2.destroyWindow(wname)


if __name__=='__main__':
    import sys
    cc = COCO(sys.argv[1])
    typeIds = [int(sys.argv[2])] if len(sys.argv) > 2 else []
    imids = list(map(int, sys.argv[3].split(','))) if len(sys.argv) > 3 else []
    cc.showImgs(imgIds=imids, typeIds=typeIds) 
