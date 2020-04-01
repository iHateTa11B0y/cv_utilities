# Utilities for computer vision tasks

This repo aim to provide some handy utilities for computer vision tasks.

## Install
```
$ pip install cvuts
```

## Utilities

### *cvuts.image_loader*
#### *ImageLoader (object)*
This is a class that can help you load image with a local path or a url. You can choose to use `PIL` or `OPENCV` by argument `core` and choose the order of the channel `RGB` or `BGR` by argument channel_order. 

### *cvuts.color_utils*
#### *get_color (function)*
Return a sequence of distinguishable color.

### *cvuts.vis_utils*
#### *vis_one_image_opencv (function)*
Show an image with bboxes, masks, tags.

### *cvuts.coco_utils*
#### *mask2poly, poly2rle, poly2rle, poly2mask, cvpoly2mask (functions)*
Provides convertion between polygon, bitmask and RLE format mask.
