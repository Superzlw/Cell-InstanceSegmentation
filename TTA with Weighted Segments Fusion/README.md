# Introduction

Most ML learner are familiar with Weighted Boxes Fusion in object detection. What about Weighted Segments Fusion (WSF) for use in instance segmentation? There are papers on the topic, but not much code, here gives a try. There are many possible approaches to this problem; take on it is a two-stage process: Start with WBF on the bounding boxes, and then fuse the segments based on the WBF output. WSF can of course used for ensembling models too, not only TTA(Test time augmentation).

References:

- [Create COCO annotations for Sartorius dataset](https://www.kaggle.com/mistag/sartorius-create-coco-annotations)
- [Cell shape analysis](https://www.kaggle.com/mistag/sartorius-cell-shape-analysis)
- [Trained CenterMask2 model](https://www.kaggle.com/mistag/train-sartorius-detectron2-centermask2)
- [Competition Metric : mAP IoU](https://www.kaggle.com/theoviel/competition-metric-map-iou)
- [Weighted boxes fusion](https://github.com/ZFTurbo/Weighted-Boxes-Fusion) by [ZFTurbo](https://kaggle.com/zfturbo)