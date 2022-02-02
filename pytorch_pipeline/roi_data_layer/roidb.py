"""Transform a roidb into a trainable roidb by adding a bunch of metadata."""

from cell_is import datasets
import numpy as np
from cell_is.model.utils.config import cfg
from cell_is.datasets.factory import get_imdb
import PIL
import pdb


def prepare_roidb(imdb):
    """Enrich the imdb's roidb by adding some derived quantities that
    are useful for training. This function precomputes the maximum
    overlap, taken over ground-truth boxes, between each ROI and
    each ground-truth box. The class with maximum overlap is also
    recorded.
    """
    # use the imdb's method roidb, which e.g., can be self.gt_roidb of imdb, to return the database of ground-truth
    # regions of interest in form of the list of dicts (keys: 'width', 'height', 'boxes', 'gt_classes', 'gt_overlaps',
    # 'flipped', 'seg_areas', 'mask')
    # the roidb is a list of dicts, whose length equals to the length of image indices
    roidb = imdb.roidb
    # TODO it should be activated when the given image size of dataset not related with coco are actually various
    # if not (imdb.name.startswith('coco')):
    #     sizes = [PIL.Image.open(imdb.image_path_at(i)).size
    #              for i in range(imdb.num_images)]

    # traverse through all images indices
    for i in range(len(imdb.image_index)):
        # XU: add two more keys with values to the 'roidb', i.e., 'img_id' and 'image' (standing for image file path)
        roidb[i]['img_id'] = imdb.image_id_at(i)
        roidb[i]['image'] = imdb.image_path_at(i)
        # TODO it should be activated when the given image size of dataset not related with coco are actually various
        # if not (imdb.name.startswith('coco')):
        #     roidb[i]['width'] = sizes[i][0]
        #     roidb[i]['height'] = sizes[i][1]
        # need gt_overlaps as a dense array for argmax
        gt_overlaps = roidb[i]['gt_overlaps']
        # max overlap with gt over classes (columns)
        max_overlaps = gt_overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = gt_overlaps.argmax(axis=1)
        # XU: add two more keys with values to the 'roidb', i.e.,'max_classes' and 'max_overlaps'
        # TODO to get more clear with these two additional columns
        roidb[i]['max_classes'] = max_classes
        roidb[i]['max_overlaps'] = max_overlaps
        # sanity checks
        # max overlap of 0 => class should be zero (background)
        zero_inds = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_inds] == 0)
        # max overlap > 0 => class should not be zero (must be a fg class)
        nonzero_inds = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_inds] != 0)


def rank_roidb_ratio(roidb):
    # rank roidb based on the ratio between width and height.
    ratio_large = 2  # largest ratio to preserve.
    ratio_small = 0.5  # smallest ratio to preserve.

    ratio_list = []
    # traverse through the len of roidb and appending its results (i.e., the quotient of width and height of the corresponding
    # image, not the bbox) into the ratio_list with the requirement that the quotient is 0.5~2
    for i in range(len(roidb)):
        width = roidb[i]['width']
        height = roidb[i]['height']
        ratio = width / float(height)

        if cfg.TRAIN.ASPECT_CROPPING:
            if ratio > ratio_large:
                roidb[i]['need_crop'] = 1
                ratio = ratio_large
            elif ratio < ratio_small:
                roidb[i]['need_crop'] = 1
                ratio = ratio_small
            else:
                roidb[i]['need_crop'] = 0
        else:
            roidb[i]['need_crop'] = 0

        ratio_list.append(ratio)

    # return the ratio_list in np.ndarray and ratio_index in np.ndarray populated by the indices corresponding to the
    # ratio values in ascending order, e.g. ratio_list: [0.5, 0.5, 1.2, 1.2, 1.2, 2, 2]
    ratio_list = np.array(ratio_list)
    ratio_index = np.argsort(ratio_list)
    return ratio_list[ratio_index], ratio_index


def filter_roidb(roidb):
    # filter the image without bounding box.
    print('before filtering, there are %d images...' % (len(roidb)))
    i = 0
    while i < len(roidb):
        if len(roidb[i]['boxes']) == 0:
            del roidb[i]
            i -= 1
        i += 1

    print('after filtering, there are %d images...' % (len(roidb)))
    return roidb


def combined_roidb(imdb_names, training=True):
    """
    Combine multiple roidbs
    """

    def get_training_roidb(imdb):
        """Returns a roidb (Region of Interest database) for use in training."""
        # the following off-line image augmentation is deprecated, instead online one would be adopted.
        # if cfg.TRAIN.USE_FLIPPED:
        #     print('Appending horizontally-flipped training examples...')
        #     imdb.append_flipped_images()
        #     print('done')

        print('Preparing training data...')

        prepare_roidb(imdb)
        # ratio_index = rank_roidb_ratio(imdb)
        print('done')

        return imdb.roidb

    def get_roidb(imdb_name):
        # XU: imdb is an instance of a specified class ,e.g. coco: coco('train', '2014')
        imdb = get_imdb(imdb_name)
        # XU: Say that in the case of using coco -> 'coco_' + year + '_' + image_set
        print('Loaded dataset `{:s}` for training'.format(imdb.name))
        # XU: Use set_proposal_method method of imdb (instance of class coco), which inherits from the parent class imdb,
        # to set the parent's attribute (self._roidb_handler) as the method of imdb - self.gt_roidb
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)  # cfg.TRAIN.PROPOSAL_METHOD (by default) = 'gt'
        print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
        roidb = get_training_roidb(imdb)
        return roidb

    # e.g., "coco_2014_train+coco_2014_valminusminival" or "livecell_coco_train"
    # XU: to return a list of lists of dicts or a list of a list of dicts if '+' is not included
    print([s for s in imdb_names.split('+')])
    roidbs = [get_roidb(s) for s in imdb_names.split('+')]
    roidb = roidbs[0]

    if len(roidbs) > 1:
        # XU: to extend all the dicts of the other roidb to the end of the first roidb
        # XU: so that it results in roidb as a list of dicts, each dict being related with one image
        for r in roidbs[1:]:
            roidb.extend(r)
        tmp = get_imdb(imdb_names.split('+')[1])
        imdb = datasets.imdb.imdb(imdb_names, tmp.classes)
    else:
        imdb = get_imdb(imdb_names)

    if training:
        roidb = filter_roidb(roidb)

    ratio_list, ratio_index = rank_roidb_ratio(roidb)

    return imdb, roidb, ratio_list, ratio_index

if __name__ == '__main__':
    import sys
    ut_imdb_name = "livecell_coco_val"
    ut_imdb, ut_roidb, ut_ratio_list, ut_ratio_index = combined_roidb(ut_imdb_name)
    for k, j in enumerate(ut_ratio_index):
        if j == 149:
            print(k)
            break
    print(f'the size of ut_roidb is: {sys.getsizeof(ut_roidb) / (1024**2)} Mb')
    import pprint
    pprint.pprint(ut_roidb[0])
