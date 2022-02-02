# --------------------------------------------------------
# Reference: Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

from cell_is.datasets.coco import coco
from cell_is.datasets.livecell import livecell
from cell_is.datasets.competition import competition

import numpy as np


# Set up coco_2014_<split>
for year in ['2014']:
    for split in ['train', 'val', 'minival', 'valminusminival', 'trainval']:
        name = f'coco_{year}_{split}'
        __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
    for split in ['test', 'test-dev']:
        name = f'coco_{year}_{split}'
        __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up livecell_coco_<split>
for genre in ['coco']:
    for split in ['train', 'val']:
        name = f'livecell_{genre}_{split}'
        __sets[name] = (lambda split=split, genre=genre: livecell(split, genre))

# Set up livecell_coco_<split>
for genre in ['coco']:
    for split in ['test']:
        name = f'livecell_{genre}_{split}'
        __sets[name] = (lambda split=split, genre=genre: livecell(split, genre))

# Setup competition_2021_<split>, where 'train' represents the train+val dataset
# TODO (XU) to consider the split!
for year in ['2021']:
    for split in ['train']:
        name = f'competition_{year}_{split}'
        __sets[name] = (lambda split=split, year=year: competition(split, year))

# Setup competition_2021_<split>
# TODO (XU) to consider the split!
for year in ['2021']:
    for split in ['test']:
        name = f'competition_{year}_{split}'
        __sets[name] = (lambda split=split, year=year: competition(split, year))


def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
