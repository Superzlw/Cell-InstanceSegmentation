# --------------------------------------------------------
# Reference: Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

from pytorch_pipeline.datasets.imdb import imdb
from pytorch_pipeline.datasets import ds_utils as ds_utils
from pytorch_pipeline.model.utils.config import cfg
import os.path as osp
import sys
import os
import numpy as np
import scipy.sparse
import scipy.io as sio
import pickle
import json
import uuid
# COCO API
from pytorch_pipeline.pycocotools.coco import COCO
from pytorch_pipeline.pycocotools.cocoeval import COCOeval
from pytorch_pipeline.pycocotools import mask as COCOmask

class competition(imdb):
  def __init__(self, image_set, year):
    imdb.__init__(self, 'competition_' + year + '_' + image_set)
    # COCO specific config options
    self.config = {'use_salt': True,
                   'cleanup': True}
    # name, paths
    self._year = year
    self._image_set = image_set
    self._data_path = osp.join(cfg.DATA_DIR, 'coco')
    # load COCO API, classes, class <-> id mappings
    self._COCO = COCO(self._get_ann_file())  # load annotations into memory
    # XU: first get the all category ids (e.g., 1~20 for twenty classes without background ) in list format given in the
    # categories part of the dataset OR get the only category id (1 for the only cell class) in list format in our case
    # (LIVECell) and then should return the category contents corresponding to the extracted category ids above in list
    # format
    cats = self._COCO.loadCats(self._COCO.getCatIds())
    # XU: return tuple: ('__background__', 'cat', 'dog', ...), which then overrides the original self._classes in class
    # imdb
    self._classes = tuple(['__background__'] + [c['name'] for c in cats])
    # XU: return dict of key for category name including __background__ and value for number ranging from 0 to including
    # len(num_classes)-1
    self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
    # XU: return dict of key for category without __background__ name and value for number ranging from 1 to including
    # len(num_classes)-1
    self._class_to_coco_cat_id = dict(list(zip([c['name'] for c in cats],
                                               self._COCO.getCatIds())))
    # XU: return the image ids of the images part of the dataset
    self._image_index = self._load_image_set_index()
    # Default to roidb handler
    self.set_proposal_method('gt')
    self.competition_mode(False)

    # Some image sets are "views" (i.e. subsets) into others.
    # For example, minival2014 is a random 5000 image subset of val2014.
    # This mapping tells us where the view's images and proposals come from.
    self._view_map = {
      'minival2014': 'val2014',  # 5k val2014 subset
      'valminusminival2014': 'val2014',  # val2014 \setminus minival2014
      'test-dev2015': 'test2015',
      'valminuscapval2014': 'val2014',
      'capval2014': 'val2014',
      'captest2014': 'val2014'
    }
    coco_name = image_set + year  # e.g., "val2014"
    self._data_name = (self._view_map[coco_name]
                       if coco_name in self._view_map
                       else coco_name)
    # Dataset splits that have ground-truth annotations (test splits
    # do not have gt annotations)
    self._gt_splits = ('train', 'val', 'minival')

  def _get_ann_file(self):
    # to get the annotation file
    # if the string 'test' is not found within it, then return e.g. 'data/coco/annotations/instances_train2014.json'
    # else return e.g. 'data/coco/image_info_test2014.json'
    prefix = 'instances' if self._image_set.find('test') == -1 \
      else 'image_info'
    return osp.join(self._data_path, 'annotations',
                    prefix + '_' + self._image_set + self._year + '.json')

  def _load_image_set_index(self):
    """
    Load image ids.
    """
    image_ids = self._COCO.getImgIds()
    return image_ids

  def _get_widths(self):
    anns = self._COCO.loadImgs(self._image_index)
    widths = [ann['width'] for ann in anns]
    return widths

  def image_path_at(self, i):
    """
    Return the absolute path to image i in the image sequence.
    """
    return self.image_path_from_index(self._image_index[i])

  def image_id_at(self, i):
    """
    Return the absolute path to image i in the image sequence.
    """
    return self._image_index[i]

  def image_path_from_index(self, index):
    """
    Construct an image path from the image's "index" identifier.
    """
    # Example image path for index=119993:
    #   images/train2014/COCO_train2014_000000119993.jpg
    file_name = ('COCO_' + self._data_name + '_' +
                 str(index).zfill(12) + '.jpg')
    image_path = osp.join(self._data_path, 'images',
                          self._data_name, file_name)
    assert osp.exists(image_path), \
      'Path does not exist: {}'.format(image_path)
    return image_path

  def gt_roidb(self):
    """
    Return the database of ground-truth regions of interest.
    This function loads/saves from/to a cache file to speed up future calls.
    """
    cache_file = osp.join(self.cache_path, self.name + '_gt_roidb.pkl')
    if osp.exists(cache_file):
      with open(cache_file, 'rb') as fid:
        roidb = pickle.load(fid)
      print('{} gt roidb loaded from {}'.format(self.name, cache_file))
      return roidb

    gt_roidb = [self._load_coco_annotation(index)
                for index in self._image_index]

    with open(cache_file, 'wb') as fid:
      pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
    print('wrote gt roidb to {}'.format(cache_file))
    return gt_roidb

  def _load_coco_annotation(self, index):
    """
    Loads COCO bounding-box instance annotations. Crowd instances are
    handled by marking their overlaps (with all categories) to -1. This
    overlap value means that crowd "instances" are excluded from training.
    """
    # XU: return the image annotation, which is a dict after slicing the first element, specific to the given index
    im_ann = self._COCO.loadImgs(index)[0]
    width = im_ann['width']
    height = im_ann['height']

    # return the annotation ids (i.e., list of ann ids) for the given image id
    annIds = self._COCO.getAnnIds(imgIds=index, iscrowd=None)
    # return the list of annotations, each of which correspond to the resulted annotation id above
    objs = self._COCO.loadAnns(annIds)
    # Sanitize bboxes -- some are invalid
    valid_objs = []
    for obj in objs:
      x1 = np.max((0, obj['bbox'][0]))
      y1 = np.max((0, obj['bbox'][1]))
      x2 = np.min((width - 1, x1 + np.max((0, obj['bbox'][2] - 1))))
      y2 = np.min((height - 1, y1 + np.max((0, obj['bbox'][3] - 1))))
      if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
        obj['clean_bbox'] = [x1, y1, x2, y2]
        valid_objs.append(obj)
    objs = valid_objs
    # return the length of the resulting valid objects that only own the valid bounding boxes
    num_objs = len(objs)

    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
    seg_areas = np.zeros((num_objs), dtype=np.float32)
    instance_masks = []

    # Lookup table to map from COCO category ids to our internal class
    # indices
    coco_cat_id_to_class_ind = dict([(self._class_to_coco_cat_id[cls],
                                      self._class_to_ind[cls])
                                     for cls in self._classes[1:]])

    for ix, obj in enumerate(objs):
      cls = coco_cat_id_to_class_ind[obj['category_id']]
      boxes[ix, :] = obj['clean_bbox']
      gt_classes[ix] = cls
      seg_areas[ix] = obj['area']
      if obj['iscrowd']:
        # Set overlap to -1 for all classes for crowd objects
        # so they will be excluded during training
        overlaps[ix, :] = -1.0
      else:
        overlaps[ix, cls] = 1.0

      # Build mask of shape [height, width, instance_count]
      if cls:
        m = self.annToMask(obj, height, width)
        # Some objects are so small that they're less than 1 pixel area
        # and end up rounded out. Skip those objects.
        if m.max() < 1:
          continue
        # Is it a crowd? If so, use a negative class ID.
        if obj['iscrowd']:
          # Use negative class ID for crowds
          cls *= -1
          # For crowd masks, annToMask() sometimes returns a mask
          # smaller than the given dimensions. If so, resize it.
          if m.shape[0] != height or m.shape[1] != width:
            m = np.ones((height, width), dtype=bool)
        instance_masks.append(m)
    mask = np.stack(instance_masks, axis=2).astype(np.bool)

    ds_utils.validate_boxes(boxes, width=width, height=height)
    overlaps = scipy.sparse.csr_matrix(overlaps)
    return {'width': width,
            'height': height,
            'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_overlaps': overlaps,
            'flipped': False,
            'seg_areas': seg_areas}

  def annToRLE(self, ann, height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    :return: binary mask (numpy 2D array)
    """
    segm = ann['segmentation']
    if isinstance(segm, list):
      # polygon -- a single object might consist of multiple parts
      # we merge all parts into one mask rle code
      rles = COCOmask.frPyObjects(segm, height, width)
      rle = COCOmask.merge(rles)
    elif isinstance(segm['counts'], list):
      # uncompressed RLE
      rle = COCOmask.frPyObjects(segm, height, width)
    else:
      # rle
      rle = ann['segmentation']
    return rle

  def annToMask(self, ann, height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
    :return: binary mask (numpy 2D array)
    """
    rle = self.annToRLE(ann, height, width)
    m = COCOmask.decode(rle)
    return m

  def _get_widths(self):
    return [r['width'] for r in self.roidb]

  def append_flipped_images(self):
    num_images = self.num_images
    widths = self._get_widths()
    for i in range(num_images):
      boxes = self.roidb[i]['boxes'].copy()
      oldx1 = boxes[:, 0].copy()
      oldx2 = boxes[:, 2].copy()
      boxes[:, 0] = widths[i] - oldx2 - 1
      boxes[:, 2] = widths[i] - oldx1 - 1
      assert (boxes[:, 2] >= boxes[:, 0]).all()
      entry = {'width': widths[i],
               'height': self.roidb[i]['height'],
               'boxes': boxes,
               'gt_classes': self.roidb[i]['gt_classes'],
               'gt_overlaps': self.roidb[i]['gt_overlaps'],
               'flipped': True,
               'seg_areas': self.roidb[i]['seg_areas']}

      self.roidb.append(entry)
    self._image_index = self._image_index * 2

  def _get_box_file(self, index):
    # first 14 chars / first 22 chars / all chars + .mat
    # COCO_val2014_0/COCO_val2014_000000447/COCO_val2014_000000447991.mat
    file_name = ('COCO_' + self._data_name +
                 '_' + str(index).zfill(12) + '.mat')
    return osp.join(file_name[:14], file_name[:22], file_name)

  def _print_detection_eval_metrics(self, coco_eval):
    IoU_lo_thresh = 0.5
    IoU_hi_thresh = 0.95

    def _get_thr_ind(coco_eval, thr):
      ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                     (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
      iou_thr = coco_eval.params.iouThrs[ind]
      assert np.isclose(iou_thr, thr)
      return ind

    ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
    ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)
    # precision has dims (iou, recall, cls, area range, max dets)
    # area range index 0: all area ranges
    # max dets index 2: 100 per image
    precision = \
      coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
    ap_default = np.mean(precision[precision > -1])
    print(('~~~~ Mean and per-category AP @ IoU=[{:.2f},{:.2f}] '
           '~~~~').format(IoU_lo_thresh, IoU_hi_thresh))
    print('{:.1f}'.format(100 * ap_default))
    for cls_ind, cls in enumerate(self.classes):
      if cls == '__background__':
        continue
      # minus 1 because of __background__
      precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, cls_ind - 1, 0, 2]
      ap = np.mean(precision[precision > -1])
      print('{:.1f}'.format(100 * ap))

    print('~~~~ Summary metrics ~~~~')
    coco_eval.summarize()

  def _do_detection_eval(self, res_file, output_dir):
    ann_type = 'bbox'
    coco_dt = self._COCO.loadRes(res_file)
    coco_eval = COCOeval(self._COCO, coco_dt)
    coco_eval.params.useSegm = (ann_type == 'segm')
    coco_eval.evaluate()
    coco_eval.accumulate()
    self._print_detection_eval_metrics(coco_eval)
    eval_file = osp.join(output_dir, 'detection_results.pkl')
    with open(eval_file, 'wb') as fid:
      pickle.dump(coco_eval, fid, pickle.HIGHEST_PROTOCOL)
    print('Wrote COCO eval results to: {}'.format(eval_file))

  def _coco_results_one_category(self, boxes, cat_id):
    results = []
    for im_ind, index in enumerate(self.image_index):
      dets = boxes[im_ind].astype(np.float)
      if dets == []:
        continue
      scores = dets[:, -1]
      xs = dets[:, 0]
      ys = dets[:, 1]
      ws = dets[:, 2] - xs + 1
      hs = dets[:, 3] - ys + 1
      results.extend(
        [{'image_id': index,
          'category_id': cat_id,
          'bbox': [xs[k], ys[k], ws[k], hs[k]],
          'score': scores[k]} for k in range(dets.shape[0])])
    return results

  def _write_coco_results_file(self, all_boxes, res_file):
    # [{"image_id": 42,
    #   "category_id": 18,
    #   "bbox": [258.15,41.29,348.26,243.78],
    #   "score": 0.236}, ...]
    results = []
    for cls_ind, cls in enumerate(self.classes):
      if cls == '__background__':
        continue
      print('Collecting {} results ({:d}/{:d})'.format(cls, cls_ind,
                                                       self.num_classes - 1))
      coco_cat_id = self._class_to_coco_cat_id[cls]
      results.extend(self._coco_results_one_category(all_boxes[cls_ind],
                                                     coco_cat_id))
    print('Writing results json to {}'.format(res_file))
    with open(res_file, 'w') as fid:
      json.dump(results, fid)

  def evaluate_detections(self, all_boxes, output_dir):
    res_file = osp.join(output_dir, ('detections_' +
                                     self._image_set +
                                     self._year +
                                     '_results'))
    if self.config['use_salt']:
      res_file += '_{}'.format(str(uuid.uuid4()))
    res_file += '.json'
    self._write_coco_results_file(all_boxes, res_file)
    # Only do evaluation on non-test sets
    if self._image_set.find('test') == -1:
      self._do_detection_eval(res_file, output_dir)
    # Optionally cleanup results json file
    if self.config['cleanup']:
      os.remove(res_file)

  def competition_mode(self, on):
    if on:
      self.config['use_salt'] = False
      self.config['cleanup'] = False
    else:
      self.config['use_salt'] = True
      self.config['cleanup'] = True
