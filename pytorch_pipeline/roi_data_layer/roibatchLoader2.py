
# """The data layer used during training to train a Fast R-CNN network.
# """

# import torch.utils.data as data
# from PIL import Image
# import torch
# from torch.utils.data.sampler import Sampler

# from utils.config import cfg
# from roi_data_layer.minibatch import get_minibatch
# #from rkx_cell_is.cell_is.model.rpn.bbox_transform import bbox_transform_inv, clip_boxes

# import numpy as np
# import random
# import time
# import pdb


# class sampler(Sampler):
#     def __init__(self, train_size, batch_size):
#         # XU: batch_size specifies the number of a batch of images
#         num_data = train_size
#         self.num_data = num_data
#         self.num_per_batch = int(num_data / batch_size)
#         self.batch_size = batch_size
#         self.range = torch.arange(0, batch_size).view(1, batch_size).long()
#         self.leftover_flag = False
#         if num_data % batch_size:
#             self.leftover = torch.arange(self.num_per_batch * batch_size, num_data).long()
#             self.leftover_flag = True

#     def __iter__(self):
#         rand_num = torch.randperm(self.num_per_batch).view(-1, 1) * self.batch_size
#         # rand_num = torch.arange(self.num_per_batch).long().view(-1, 1) * self.batch_size
#         self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range
#         # to flatten the tensor array
#         self.rand_num_view = self.rand_num.view(-1)

#         if self.leftover_flag:
#             self.rand_num_view = torch.cat((self.rand_num_view, self.leftover), 0)

#         return iter(self.rand_num_view)

#     def __len__(self):
#         return self.num_data


# class roibatchLoader(data.Dataset):
#   def __init__(self, imdb, roidb, ratio_list, ratio_index, batch_size, num_classes, training=True, normalize=None):
#     self._imdb = imdb
#     self._roidb = roidb
#     self._num_classes = num_classes  # 2 in the case of livecell
#     # we make the height of image consistent to trim_height, trim_width
#     self.trim_height = cfg.TRAIN.TRIM_HEIGHT
#     self.trim_width = cfg.TRAIN.TRIM_WIDTH
#     # TODO (XU) the configuring MAX_NUM_GT_BOXES should be considered in terms of its size
#     self.max_num_box = cfg.MAX_NUM_GT_BOXES
#     self.training = training
#     self.normalize = normalize
#     self.ratio_list = ratio_list
#     self.ratio_index = ratio_index
#     self.batch_size = batch_size
#     # XU: the number of images
#     self.data_size = len(self.ratio_list)

#     # given the ratio_list, we want to make the ratio same for each batch.
#     # XU: all the ratios in self.ratio_list_batch are the same in our case = 704 / float(520)
#     self.ratio_list_batch = torch.Tensor(self.data_size).zero_()
#     # XU: return the number of batches which should contain the rest batch whose sample number is < batch_size if any
#     num_batch = int(np.ceil(len(ratio_index) / batch_size))
#     for i in range(num_batch):
#         left_idx = i*batch_size
#         right_idx = min((i+1)*batch_size-1, self.data_size-1)

#         if ratio_list[right_idx] < 1:
#             # for ratio < 1, we preserve the leftmost in each batch.
#             target_ratio = ratio_list[left_idx]
#         elif ratio_list[left_idx] > 1:
#             # for ratio > 1, we preserve the rightmost in each batch.
#             target_ratio = ratio_list[right_idx]
#         else:
#             # for ratio cross 1, we make it to be 1.
#             target_ratio = 1

#         self.ratio_list_batch[left_idx:(right_idx+1)] = target_ratio

#   def __getitem__(self, index):
#     if self.training:
#         # XU: the index_ratio is the index sampled from the ndarray self.ratio_index
#         index_ratio = int(self.ratio_index[index])
#     else:
#         index_ratio = index

#     # get the anchor index for current sample index
#     # here we set the anchor index to the last one
#     # sample in this group
#     # XU: return a list of a dict, the only dict being related with one image
#     minibatch_db = [self._roidb[index_ratio]]
#     # XU: return a dict of keys - ['data', 'gt_boxes', 'im_info', 'img_id', 'gt_masks'] and values - ...
#     blobs = get_minibatch(self._imdb, minibatch_db, self._num_classes)
#     data = torch.from_numpy(blobs['data'])
#     im_info = torch.from_numpy(blobs['im_info'])  # shape: (1,3)
#     # we need to random shuffle the bounding box.
#     data_height, data_width = data.size(dim=1), data.size(dim=2)
#     if self.training:
#         blobs_ids = np.arange(len(blobs['gt_boxes']))
#         np.random.shuffle(blobs_ids)
#         gt_boxes = torch.from_numpy(blobs['gt_boxes'][blobs_ids])
#         gt_masks = torch.from_numpy(blobs['gt_masks'][:, :, blobs_ids])
#         # if self.batch_size == 1:
#         #     data = data.permute(0, 3, 1, 2).contiguous().view(3, data_height, data_width)
#         #     im_info = im_info.view(3)
#         #     num_boxes = gt_boxes.size(0)
#         #
#         #     return data, im_info, gt_boxes, num_boxes, blobs['img_id']

#         ########################################################
#         # padding the input image to fixed size for each group #
#         ########################################################

#         # NOTE1: need to cope with the case where a group cover both conditions. (done)
#         # NOTE2: need to consider the situation for the tail samples. (no worry)
#         # NOTE3: need to implement a parallel data loader. (no worry)
#         # get the index range

#         # if the image need to crop, crop to the target size.
#         ratio = self.ratio_list_batch[index]

#         # XU: the following code block can be removed since all images of livecell are of the same size
#         # XU: and have ratio ~[0.5, 2]
#         if self._roidb[index_ratio]['need_crop']:
#             if ratio < 1:
#                 # this means that data_width << data_height, we need to crop the
#                 # data_height
#                 min_y = int(torch.min(gt_boxes[:,1]))
#                 max_y = int(torch.max(gt_boxes[:,3]))
#                 trim_size = int(np.floor(data_width / ratio))
#                 box_region = max_y - min_y + 1
#                 if min_y == 0:
#                     y_s = 0
#                 else:
#                     if (box_region-trim_size) < 0:
#                         y_s_min = max(max_y-trim_size, 0)
#                         y_s_max = min(min_y, data_height-trim_size)
#                         if y_s_min == y_s_max:
#                             y_s = y_s_min
#                         else:
#                             y_s = np.random.choice(range(y_s_min, y_s_max))
#                     else:
#                         y_s_add = int((box_region-trim_size)/2)
#                         if y_s_add == 0:
#                             y_s = min_y
#                         else:
#                             y_s = np.random.choice(range(min_y, min_y+y_s_add))
#                 # crop the image
#                 data = data[:, y_s:(y_s + trim_size), :, :]

#                 # shift y coordiante of gt_boxes
#                 gt_boxes[:, 1] = gt_boxes[:, 1] - float(y_s)
#                 gt_boxes[:, 3] = gt_boxes[:, 3] - float(y_s)

#                 # update gt bounding box according the trip
#                 gt_boxes[:, 1].clamp_(0, trim_size - 1)
#                 gt_boxes[:, 3].clamp_(0, trim_size - 1)

#             else:
#                 # this means that data_width >> data_height, we need to crop the
#                 # data_width
#                 min_x = int(torch.min(gt_boxes[:,0]))
#                 max_x = int(torch.max(gt_boxes[:,2]))
#                 trim_size = int(np.ceil(data_height * ratio))
#                 box_region = max_x - min_x + 1
#                 if min_x == 0:
#                     x_s = 0
#                 else:
#                     if (box_region-trim_size) < 0:
#                         x_s_min = max(max_x-trim_size, 0)
#                         x_s_max = min(min_x, data_width-trim_size)
#                         if x_s_min == x_s_max:
#                             x_s = x_s_min
#                         else:
#                             x_s = np.random.choice(range(x_s_min, x_s_max))
#                     else:
#                         x_s_add = int((box_region-trim_size)/2)
#                         if x_s_add == 0:
#                             x_s = min_x
#                         else:
#                             x_s = np.random.choice(range(min_x, min_x+x_s_add))
#                 # crop the image
#                 data = data[:, :, x_s:(x_s + trim_size), :]

#                 # shift x coordiante of gt_boxes
#                 gt_boxes[:, 0] = gt_boxes[:, 0] - float(x_s)
#                 gt_boxes[:, 2] = gt_boxes[:, 2] - float(x_s)
#                 # update gt bounding box according the trip
#                 gt_boxes[:, 0].clamp_(0, trim_size - 1)
#                 gt_boxes[:, 2].clamp_(0, trim_size - 1)

#         # based on the ratio, padding the image.
#         if ratio < 1:
#             # this means that data_width < data_height
#             trim_size = int(np.floor(data_width / ratio))

#             padding_data = torch.FloatTensor(int(np.ceil(data_width / ratio)),
#                                              data_width, 3).zero_()

#             padding_data[:data_height, :, :] = data[0]
#             # update im_info
#             im_info[0, 0] = padding_data.size(0)
#             # print("height %d %d \n" %(index, anchor_idx))
#         elif ratio > 1:
#             # this means that data_width > data_height
#             # if the image need to crop.
#             padding_data = torch.FloatTensor(data_height,
#                                              int(np.ceil(data_height * ratio)), 3).zero_()
#             padding_data[:, :data_width, :] = data[0]  # shape: (h, w, 3)
#             im_info[0, 1] = padding_data.size(1)
#         else:
#             trim_size = min(data_height, data_width)
#             padding_data = torch.FloatTensor(trim_size, trim_size, 3).zero_()
#             padding_data = data[0][:trim_size, :trim_size, :]
#             gt_boxes.clamp_(0, trim_size)
#             im_info[0, 0] = trim_size
#             im_info[0, 1] = trim_size


#         # check the bounding box:
#         # TODO (XU) to possibly add the part of minimizing the mask to reduce memory overload
#         not_keep = (gt_boxes[:, 0] == gt_boxes[:, 2]) | (gt_boxes[:, 1] == gt_boxes[:, 3])
#         keep = torch.nonzero(not_keep == 0).view(-1)

#         gt_boxes_padding = torch.FloatTensor(self.max_num_box,
#                                              gt_boxes.size(dim=1)).zero_()  # shape: (self.max_num_box, 5)
#         gt_masks_padding = torch.ByteTensor(gt_masks.size(dim=0), gt_masks.size(dim=1),
#                                             self.max_num_box).zero_()  # shape: (h, w, self.max_num_box)
#         if keep.numel() != 0:
#             gt_boxes = gt_boxes[keep]
#             gt_masks = gt_masks[:, :, keep]
#             num_boxes = min(gt_boxes.size(0), self.max_num_box)
#             gt_boxes_padding[:num_boxes, :] = gt_boxes[:num_boxes]  # shape: (self.max_num_box, 5)
#             gt_masks_padding[:, :, :num_boxes] = gt_masks[:, :, :num_boxes]  # shape: (h, w, self.max_num_box)
#         else:
#             num_boxes = 0

#         # permute trim_data to adapt to downstream processing
#         padding_data = padding_data.permute(2, 0, 1).contiguous()  # shape: (3, h, w)
#         im_info = im_info.view(3)  # shape: (3,)


#         labels = torch.ones((self.data_size,), dtype=torch.int64)
        
#         target = {}
#         target["boxes"] = gt_boxes_padding
#         target["labels"] = labels
#         target["masks"] = gt_masks_padding
#         target["image_id"] = blobs['img_id']
#         target["area"] = blobs['area']
#         target["iscrowd"] = blobs['iscrowd']
#         print(padding_data.shape)
#         return padding_data, target
        
#         #return padding_data, im_info, gt_boxes_padding, gt_masks_padding, num_boxes, blobs['img_id']
#     else:
#         data = data.permute(0, 3, 1, 2).contiguous().view(3, data_height, data_width)
#         im_info = im_info.view(3)

#         gt_boxes = torch.FloatTensor([1,1,1,1,1])
#         # TODO (XU) to figure out how to build the gt_masks when not under training
#         gt_masks = torch.ByteTensor()
#         num_boxes = 0

#         return data, im_info, gt_boxes, num_boxes

#   def __len__(self):
#     return len(self._roidb)


"""The data layer used during training to train a Fast R-CNN network.
"""

import torch.utils.data as data
from PIL import Image
import torch
from torch.utils.data.sampler import Sampler

from utils.config import cfg
from roi_data_layer.minibatch import get_minibatch
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes

import numpy as np
import random
import time
import pdb


class sampler(Sampler):
    def __init__(self, train_size, batch_size):
        # XU: batch_size specifies the number of a batch of images
        num_data = train_size
        self.num_data = num_data
        self.num_per_batch = int(num_data / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0, batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if num_data % batch_size:
            self.leftover = torch.arange(self.num_per_batch * batch_size, num_data).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1, 1) * self.batch_size
        # rand_num = torch.arange(self.num_per_batch).long().view(-1, 1) * self.batch_size
        self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range
        # to flatten the tensor array
        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover), 0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data


class roibatchLoader(data.Dataset):
  def __init__(self, imdb, roidb, ratio_list, ratio_index, batch_size, num_classes, training=True, normalize=None):
    self._imdb = imdb
    self._roidb = roidb
    self._num_classes = num_classes  # 2 in the case of livecell
    # we make the height of image consistent to trim_height, trim_width
    self.trim_height = cfg.TRAIN.TRIM_HEIGHT
    self.trim_width = cfg.TRAIN.TRIM_WIDTH
    # TODO (XU) the configuring MAX_NUM_GT_BOXES should be considered in terms of its size
    self.max_num_box = cfg.MAX_NUM_GT_BOXES
    self.training = training
    self.normalize = normalize
    self.ratio_list = ratio_list
    self.ratio_index = ratio_index
    self.batch_size = batch_size
    # XU: the number of images
    self.data_size = len(self.ratio_list)

    # given the ratio_list, we want to make the ratio same for each batch.
    # XU: all the ratios in self.ratio_list_batch are the same in our case = 704 / float(520)
    self.ratio_list_batch = torch.Tensor(self.data_size).zero_()
    # XU: return the number of batches which should contain the rest batch whose sample number is < batch_size if any
    num_batch = int(np.ceil(len(ratio_index) / batch_size))
    for i in range(num_batch):
        left_idx = i*batch_size
        right_idx = min((i+1)*batch_size-1, self.data_size-1)

        if ratio_list[right_idx] < 1:
            # for ratio < 1, we preserve the leftmost in each batch.
            target_ratio = ratio_list[left_idx]
        elif ratio_list[left_idx] > 1:
            # for ratio > 1, we preserve the rightmost in each batch.
            target_ratio = ratio_list[right_idx]
        else:
            # for ratio cross 1, we make it to be 1.
            target_ratio = 1

        self.ratio_list_batch[left_idx:(right_idx+1)] = target_ratio

  def __getitem__(self, index):
    print(index)
    if self.training:
        # XU: the index_ratio is the index sampled from the ndarray self.ratio_index
        index_ratio = int(self.ratio_index[index])
    else:
        index_ratio = index

    # get the anchor index for current sample index
    # here we set the anchor index to the last one
    # sample in this group
    # XU: return a list of a dict, the only dict being related with one image
    minibatch_db = [self._roidb[index_ratio]]
    # XU: return a dict of keys - ['data', 'gt_boxes', 'im_info', 'img_id', 'gt_masks'] and values - ...
    blobs = get_minibatch(self._imdb, minibatch_db, self._num_classes)
    data = torch.from_numpy(blobs['data'])
    im_info = torch.from_numpy(blobs['im_info'])  # shape: (1,3)
    # we need to random shuffle the bounding box.
    data_height, data_width = data.size(dim=1), data.size(dim=2)
    if self.training:
        blobs_ids = np.arange(len(blobs['gt_boxes']))
        np.random.shuffle(blobs_ids)
        gt_boxes = torch.from_numpy(blobs['gt_boxes'][blobs_ids])
        gt_masks = torch.from_numpy(blobs['gt_masks'][:, :, blobs_ids])
        # if self.batch_size == 1:
        #     data = data.permute(0, 3, 1, 2).contiguous().view(3, data_height, data_width)
        #     im_info = im_info.view(3)
        #     num_boxes = gt_boxes.size(0)
        #
        #     return data, im_info, gt_boxes, num_boxes, blobs['img_id']

        ########################################################
        # padding the input image to fixed size for each group #
        ########################################################

        # NOTE1: need to cope with the case where a group cover both conditions. (done)
        # NOTE2: need to consider the situation for the tail samples. (no worry)
        # NOTE3: need to implement a parallel data loader. (no worry)
        # get the index range

        # if the image need to crop, crop to the target size.
        ratio = self.ratio_list_batch[index]

        # XU: the following code block can be removed since all images of livecell are of the same size
        # XU: and have ratio ~[0.5, 2]
        if self._roidb[index_ratio]['need_crop']:
            if ratio < 1:
                # this means that data_width << data_height, we need to crop the
                # data_height
                min_y = int(torch.min(gt_boxes[:,1]))
                max_y = int(torch.max(gt_boxes[:,3]))
                trim_size = int(np.floor(data_width / ratio))
                box_region = max_y - min_y + 1
                if min_y == 0:
                    y_s = 0
                else:
                    if (box_region-trim_size) < 0:
                        y_s_min = max(max_y-trim_size, 0)
                        y_s_max = min(min_y, data_height-trim_size)
                        if y_s_min == y_s_max:
                            y_s = y_s_min
                        else:
                            y_s = np.random.choice(range(y_s_min, y_s_max))
                    else:
                        y_s_add = int((box_region-trim_size)/2)
                        if y_s_add == 0:
                            y_s = min_y
                        else:
                            y_s = np.random.choice(range(min_y, min_y+y_s_add))
                # crop the image
                data = data[:, y_s:(y_s + trim_size), :, :]

                # shift y coordiante of gt_boxes
                gt_boxes[:, 1] = gt_boxes[:, 1] - float(y_s)
                gt_boxes[:, 3] = gt_boxes[:, 3] - float(y_s)

                # update gt bounding box according the trip
                gt_boxes[:, 1].clamp_(0, trim_size - 1)
                gt_boxes[:, 3].clamp_(0, trim_size - 1)

            else:
                # this means that data_width >> data_height, we need to crop the
                # data_width
                min_x = int(torch.min(gt_boxes[:,0]))
                max_x = int(torch.max(gt_boxes[:,2]))
                trim_size = int(np.ceil(data_height * ratio))
                box_region = max_x - min_x + 1
                if min_x == 0:
                    x_s = 0
                else:
                    if (box_region-trim_size) < 0:
                        x_s_min = max(max_x-trim_size, 0)
                        x_s_max = min(min_x, data_width-trim_size)
                        if x_s_min == x_s_max:
                            x_s = x_s_min
                        else:
                            x_s = np.random.choice(range(x_s_min, x_s_max))
                    else:
                        x_s_add = int((box_region-trim_size)/2)
                        if x_s_add == 0:
                            x_s = min_x
                        else:
                            x_s = np.random.choice(range(min_x, min_x+x_s_add))
                # crop the image
                data = data[:, :, x_s:(x_s + trim_size), :]

                # shift x coordiante of gt_boxes
                gt_boxes[:, 0] = gt_boxes[:, 0] - float(x_s)
                gt_boxes[:, 2] = gt_boxes[:, 2] - float(x_s)
                # update gt bounding box according the trip
                gt_boxes[:, 0].clamp_(0, trim_size - 1)
                gt_boxes[:, 2].clamp_(0, trim_size - 1)

        # based on the ratio, padding the image.
        if ratio < 1:
            # this means that data_width < data_height
            trim_size = int(np.floor(data_width / ratio))

            padding_data = torch.FloatTensor(int(np.ceil(data_width / ratio)),
                                             data_width, 3).zero_()

            padding_data[:data_height, :, :] = data[0]
            # update im_info
            im_info[0, 0] = padding_data.size(0)
            # print("height %d %d \n" %(index, anchor_idx))
        elif ratio > 1:
            # this means that data_width > data_height
            # if the image need to crop.
            padding_data = torch.FloatTensor(data_height,
                                             int(np.ceil(data_height * ratio)), 3).zero_()
            padding_data[:, :data_width, :] = data[0]  # shape: (h, w, 3)
            im_info[0, 1] = padding_data.size(1)
        else:
            trim_size = min(data_height, data_width)
            padding_data = torch.FloatTensor(trim_size, trim_size, 3).zero_()
            padding_data = data[0][:trim_size, :trim_size, :]
            gt_boxes.clamp_(0, trim_size)
            im_info[0, 0] = trim_size
            im_info[0, 1] = trim_size


        # check the bounding box:
        # TODO (XU) to possibly add the part of minimizing the mask to reduce memory overload
        not_keep = (gt_boxes[:, 0] == gt_boxes[:, 2]) | (gt_boxes[:, 1] == gt_boxes[:, 3])
        keep = torch.nonzero(not_keep == 0).view(-1)

        gt_boxes_padding = torch.FloatTensor(self.max_num_box,
                                             gt_boxes.size(dim=1)).zero_()  # shape: (self.max_num_box, 5)
        gt_masks_padding = torch.ByteTensor(gt_masks.size(dim=0), gt_masks.size(dim=1),
                                            self.max_num_box).zero_()  # shape: (h, w, self.max_num_box)
        if keep.numel() != 0:
            gt_boxes = gt_boxes[keep]
            gt_masks = gt_masks[:, :, keep]
            num_boxes = min(gt_boxes.size(0), self.max_num_box)
            gt_boxes_padding[:num_boxes, :] = gt_boxes[:num_boxes]  # shape: (self.max_num_box, 5)
            gt_masks_padding[:, :, :num_boxes] = gt_masks[:, :, :num_boxes]  # shape: (h, w, self.max_num_box)
        else:
            num_boxes = 0

        # permute trim_data to adapt to downstream processing
        padding_data = padding_data.permute(2, 0, 1).contiguous()  # shape: (3, h, w)
        im_info = im_info.view(3)  # shape: (3,)
        #from pycocotools.coco import COCO
        #coco = COCO()
        #ann_ids = coco.getAnnIds(imgIds=[index])
        #target = coco.loadAnns(ann_ids)
        #target = dict(image_id=index, annotations=target)
        print('TYPE: ',type(im_info))
        targets = []
        for i in range(self.max_num_box):
            target = dict()
            target['boxes'] = gt_boxes_padding[i,:4]
            target['masks'] = gt_masks_padding[:,:,i]
            target['labels'] = 1
            targets.append(target)
        #final_target = [dict(image_id=blobs['img_id'], annotations=targets)]
        return padding_data, targets
        #return padding_data, im_info, gt_boxes_padding, gt_masks_padding, num_boxes, blobs['img_id']
    else:
        data = data.permute(0, 3, 1, 2).contiguous().view(3, data_height, data_width)
        im_info = im_info.view(3)

        gt_boxes = torch.FloatTensor([1,1,1,1,1])
        # TODO (XU) to figure out how to build the gt_masks when not under training
        gt_masks = torch.ByteTensor()
        num_boxes = 0

        return data, im_info, gt_boxes, num_boxes

  def __len__(self):
    return len(self._roidb)

