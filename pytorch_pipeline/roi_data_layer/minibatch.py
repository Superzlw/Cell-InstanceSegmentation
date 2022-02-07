# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""

import numpy as np
import numpy.random as npr
# from scipy.misc import imread  # imread is deprecated from scipy 1.0 onwards, so use cv2 instead
import cv2
import scipy
import warnings
from utils.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob
import pdb


def get_minibatch(imdb, roidb, num_classes):
    """
    Given a roidb, construct a minibatch - blobs sampled from it.
    The returned blobs is a dict of keys - 'data', 'gt_boxes', 'im_info', 'img_id', 'gt_masks', 'area', 'iscrowd'
    """
    # XU: roidb is a list of a dict
    num_images = len(roidb)

    # Build mask after filtering, of shape (height, width, instance_count)
    # Adapt the other keys to be consistent
    bboxes = []
    instance_masks = []
    gt_classes = []
    overlaps = []
    seg_areas = []
    width = roidb[0]['width']
    height = roidb[0]['height']
    for ix, cls in enumerate(roidb[0]['gt_classes']):
        obj = {'segmentation': roidb[0]['segmentation'][ix]}
        m = imdb.annToMask(obj, height, width)
        bbox = roidb[0]['boxes'][ix]
        # Some objects are so small that they're less than 1 pixel area
        # and end up rounded out. Skip those objects.
        if m.max() < 1:
            continue
        # Is it a crowd? If so, use a negative class ID.
        if np.allclose(roidb[0]['gt_overlaps'][ix], -1.0):
            # Use negative class ID for crowds
            cls *= -1
            # For crowd masks, annToMask() sometimes returns a mask
            # smaller than the given dimensions. If so, resize it.
            if m.shape[0] != height or m.shape[1] != width:
                m = np.ones((height, width), dtype=bool)
        instance_masks.append(m)
        bboxes.append(bbox)
        gt_classes.append(cls)
        overlaps.append(roidb[0]['gt_overlaps'][ix])
        seg_areas.append(roidb[0]['seg_areas'][ix])
    #del roidb[0]['segmentation']
    roidb[0]['boxes'] = np.stack(bboxes, axis=0).astype(np.uint16)
    roidb[0]['mask'] = np.stack(instance_masks, axis=2).astype(np.bool)
    roidb[0]['gt_classes'] = np.asarray(gt_classes, dtype=np.int32)
    roidb[0]['gt_overlaps'] = np.stack(overlaps, axis=0).astype(np.float32)
    roidb[0]['seg_areas'] = np.asarray(seg_areas, dtype=np.float32)

    # Sample random scales to use for each image in this batch
    # XU: return np.ndarray of shape (num_images,) with int number uniformly sampled from [0,len(cfg.TRAIN.SCALES))
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES), size=num_images)
    # XU: cfg.TRAIN.BATCH_SIZE specifies the size of a batch of regions of interest
    assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({}) integerly'. \
        format(num_images, cfg.TRAIN.BATCH_SIZE)

    # XU: Image Augmentation: get the bboxes and masks flipped accordingly if the image would be flipped
    coin = npr.randint(0, 1, size=1, dtype=bool)[0]
    if cfg.TRAIN.USE_FLIPPED:
        if coin:
            fliplr_image_annots(roidb[0])

    # Get the input image blob, formatted for caffe
    # XU: im_blob (np.ndarray) currently has the shape: (1, h, w, 3); im_scales: a list of only one float number
    im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

    blobs = {'data': im_blob}

    assert len(im_scales) == 1, "Single batch only"  # single batch means batch of size: one
    assert len(roidb) == 1, "Single batch only"
  
    # gt boxes: (x1, y1, x2, y2, cls)
    if cfg.TRAIN.USE_ALL_GT:
        # Include all ground truth boxes
        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
    else:
        # For the COCO ground truth boxes, exclude the ones that are ''iscrowd''
        gt_inds = np.where(roidb[0]['gt_classes'] != 0 & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]
    # XU: NOTE that gt_boxes has five cols, the first four being bbox coordinates, the last being class ind
    gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
    gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
    gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
    blobs['gt_boxes'] = gt_boxes
    # XU: NOTE that roidb[0]['mask'] has shape: (h,w,instance_count)
    gt_masks_unscaled = roidb[0]['mask'][:, :, gt_inds]
    if im_scales[0] == 1:
        blobs['gt_masks'] = gt_masks_unscaled
    else:
        # TODO (XU) to add the resize_mask part after adding the resize_image part, also to be added
        # blobs['gt_masks'] = resize_mask(gt_masks_unscaled, im_scales[0], padding, crop=None)
        pass
    blobs['im_info'] = np.array(
      [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
      dtype=np.float32)
    #print(roidb[0].keys())
    blobs['img_id'] = roidb[0]['img_id']
    blobs['area'] = roidb[0]['seg_areas']
    blobs['iscrowd'] = 0

    return blobs

def _get_image_blob(roidb, scale_inds):
  """
  Builds an input blob from the image(s) in the roidb at the specified
  scales. In our case, the number of images in the blob is now 1 (which
  can be extended to batch_size). So the blob's shape is (1, h, w, 3).
  """
  num_images = len(roidb)

  processed_ims = []
  im_scales = []
  for i in range(num_images):
    # XU: BGR as default color channels, so remain it to be compatible with config
    #print(roidb)
    im = cv2.imdecode(np.fromfile(roidb[i]['image'],dtype=np.uint8),-1)
    # im = imread(roidb[i]['image'])  # imageio.imread replaces scipy.imread and it outputs RGB by default as Pillow

    if len(im.shape) == 2:  # if im.shape==3, then (h, w, c)
      im = im[:,:,np.newaxis]
      im = np.concatenate((im,im,im), axis=2)

    if roidb[i]['flipped']:
      im = im[:, ::-1, :]
    target_size = cfg.TRAIN.SCALES[scale_inds[i]]
    # XU: to standardize the image and resize the image if different target_size from size of image's shortest side is
    # XU: input to this func
    im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, cfg.PIXEL_STDS, target_size,
                    cfg.TRAIN.MAX_SIZE)
    im_scales.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, im_scales

def resize_mask(mask, scale, padding, crop=None):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the mask, are resized consistently.

    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (0, 0)]
    """
    # Suppress warning from scipy 0.13.0, the output shape of zoom() is
    # calculated with round() instead of int()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
    if crop is not None:
        y, x, h, w = crop
        mask = mask[y:y + h, x:x + w]
    else:
        mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask


def fliplr_image_annots(roidb):
    """
    flip the annotations of the image along with the flipped image horizontally
    :param roidb: dict the dict containing the annotations for the image
    :return:
    """
    width = roidb['width']
    boxes = roidb['boxes'].copy()
    oldx1 = boxes[:, 0]
    oldx2 = boxes[:, 2]
    roidb['boxes'][:, 0] = width - oldx2 - 1
    roidb['boxes'][:, 2] = width - oldx1 - 1
    assert (roidb['boxes'][:, 2] >= roidb['boxes'][:, 0]).all()
    roidb['mask'] = np.fliplr(roidb['mask'])
    roidb['flipped'] = True