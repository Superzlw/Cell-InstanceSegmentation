# Note for pytorch_pipeline package

## pycocotools module
Since LIVECell dataset has its own speciality other than coco dataset, which can be found in its annotation metadata, 
that would cause problems when loading it in terms of its annotation metadata using coco API (python). As a **temporary**
solution, the corresponding codes with the snippet `self.dataset['annotations']` would all be altered to the following:
`self.dataset['annotations'].values()`.