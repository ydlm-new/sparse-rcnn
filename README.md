训练命令	python tools/train.py configs/sparse_rcnn/sparse-rcnn_r50_fpn_1x_coco.py


测试命令	python tools/test.py configs/sparse_rcnn/sparse-rcnn_r50_fpn_1x_coco.py work_dirs/sparse-rcnn_r50_fpn_1x_coco/latest.pth --cfg-options test_evaluator.type=VOCMetric test_evaluator.metric=mAP test_evaluator.eval_mode=11points
