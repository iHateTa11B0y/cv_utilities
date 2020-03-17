import cv2
import numpy as np
from color_utils import get_color
import pycocotools.mask as mask_utils

GRAY = (218, 227, 218)
WHITE = (255, 255, 255)
GREEN = (18, 127, 15)

def add_on_boxes(
        image, 
        boxes, 
        box_colors, 
        box_thick=1, 
        tags=None,
        tag_color=WHITE,
        tag_bg_colors=None,
        tag_font_scale=0.3,
        font = cv2.FONT_HERSHEY_SIMPLEX,
        ):
    for idx, (b, c) in enumerate(zip(boxes, box_colors)):
        x0, y0, w, h = b
        x1, y1 = int(x0 + w), int(y0 + h)
        x0, y0 = int(x0), int(y0)
        cv2.rectangle(image, (x0, y0), (x1, y1), c, thickness=box_thick)

        if tags is not None:
            txt, tbc = tags[idx], tag_bg_colors[idx]
            ((txt_w, txt_h), _) = cv2.getTextSize(txt, font, tag_font_scale, 1)
            back_tl = x0, y0 - int(1.3 * txt_h)
            back_br = x0 + txt_w, y0
            cv2.rectangle(image, back_tl, back_br, tbc, -1)
            txt_tl = x0, y0 - int(0.3 * txt_h)
            cv2.putText(image, txt, txt_tl, font, tag_font_scale, tag_color, lineType=cv2.LINE_AA)

    return image

def add_on_segms(image, segms, colors, alpha=0.4, show_border=True, border_thick=1, rle=False):
    img = image.astype(np.float32)
    borders = []
    for ss, c in zip(segms, colors):
        if not rle:
            bbox_mask = np.zeros(img.shape[:2], dtype=np.int8).astype(np.bool)
            for s in ss:
                mask = np.zeros(img.shape[:2], dtype=np.int8)
                ss = np.array([[s]])
                ss = ss.reshape((1,-1,2))
                ss = np.array(ss, dtype=np.int)
                mask = cv2.fillPoly(mask, ss, 255)
                borders.append(ss)
                bbox_mask |= (mask > 50).astype(np.bool)
        else:
            mask = mask_utils.decode(ss)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
            #contours = [np.squeeze(arr) for arr in contours if len(arr) > 2]
            borders.append(contours)
            bbox_mask = (mask > 0).astype(np.bool)
        img[bbox_mask] = img[bbox_mask] * (1-alpha) + np.array(c) * alpha
    if show_border:
        for s in borders:
            cv2.drawContours(img, s, -1, WHITE, border_thick, cv2.LINE_AA)
    return img.astype(np.uint8)

def vis_one_image_opencv(
        image,
        boxes,       # x,y,w,h
        segms=None,  # poly
        tags=None,
        auto_color=False,
        rle=False,
        ):
    if segms is not None and len(segms) != len(boxes):
        print('length mismath for segms and boxes.')
        raise
    if tags is not None and len(tags) != len(boxes):
        print('length mismath for scores and boxes.')
        raise
    obj_cnt = len(boxes)

    if auto_color:
        boxes_colors = get_color(obj_cnt)
        segms_colors = boxes_colors
        tags_bg_colors = boxes_colors
    else:
        boxes_colors = [GREEN] * obj_cnt
        segms_colors = [GRAY] * obj_cnt
        tags_bg_colors = [GREEN] * obj_cnt

    image = add_on_boxes(
            image, 
            boxes, 
            boxes_colors,
            box_thick=2,
            tags=tags,
            tag_color=WHITE,
            tag_bg_colors=tags_bg_colors,
            tag_font_scale=0.6
            )

    if segms is None: 
        return image

    image = add_on_segms(
            image, 
            segms,
            segms_colors,
            alpha=0.4,
            show_border=True,
            border_thick=1,
            rle=rle,
            )

    return image

if __name__=='__main__':
    import json
    from image_loader import ImageLoader
    imloader = ImageLoader('opencv', 'bgr')
    data = json.load(open('/core1/data/home/niuwenhao/training_data/training/youbao_all.json', 'r'))
    for i in data['images']:
        if int(i['id']) == 62322:
            im_data = i
    anno_data = []
    for a in data['annotations']:
        if int(a['image_id']) == 62322:
            anno_data.append(a)
    image = imloader.load(im_data['file_name'])
    boxes = [a['bbox'] for a in anno_data]
    segms = [a['segmentation'] for a in anno_data]
    tags = [str(a['type_id']) for a in anno_data]
    image_show = vis_one_image_opencv(image,boxes,segms,tags,auto_color=True,rle=True)
    cv2.imwrite('image.jpg', image_show)
    #cv2.waitKey()
