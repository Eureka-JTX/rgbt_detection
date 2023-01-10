# bash tools/dist_train.sh configs/FLIR/faster_rcnn_r50_fpn_1x_FLIR_vis_unfreeze.py 8
# bash tools/dist_train.sh configs/FLIR/faster_rcnn_r50_fpn_1x_FLIR_thermal_unfreeze.py 8

# bash tools/dist_train.sh configs/FLIR/faster_rcnn_r50_fpn_1x_FLIR_vis_unfreeze_wd5x.py 8

# bash tools/dist_train.sh configs/FLIR/faster_rcnn_r50_fpn_1x_FLIR_post_fusion_lr0p05x.py 8
# bash tools/dist_train.sh configs/FLIR/faster_rcnn_r50_fpn_1x_FLIR_post_fusion_lr0p01x.py 8
# bash tools/dist_train.sh configs/FLIR/faster_rcnn_r50_fpn_1x_FLIR_post_fusion_lr0p1x_wd0p5.py 8
# bash tools/dist_train.sh configs/FLIR/faster_rcnn_r50_fpn_1x_FLIR_post_fusion_lr0p1x_wd5x.py 8

# bash tools/dist_train.sh configs/FLIR/faster_rcnn_r50_fpn_1x_FLIR_post_fusion_class_only_unshared_res_0p7_1.py 8
# bash tools/dist_train.sh configs/FLIR/faster_rcnn_r50_fpn_1x_FLIR_post_fusion_class_only_unshared_res_0p5_linear_bias.py 8
# bash tools/dist_train.sh configs/FLIR/faster_rcnn_r50_fpn_1x_FLIR_post_fusion_class_only_unshared_res_0p5_linear_nobias.py 8
# bash tools/dist_train.sh configs/FLIR/faster_rcnn_r50_fpn_1x_FLIR_post_fusion_class_only_unshared_res_0p3_1.py 8

# bash tools/dist_train.sh configs/FLIR/faster_rcnn_r50_fpn_1x_FLIR_post_fusion_class_only_unshared_res_0p0.py 8

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,8 bash tools/dist_train_search.sh configs/FLIR/faster_rcnn_r50_fpn_1x_FLIR_search_r1.py 8
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,8 bash tools/dist_train_search.sh configs/FLIR/faster_rcnn_r50_fpn_1x_FLIR_search_r2.py 8
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,8 bash tools/dist_train_search.sh configs/FLIR/faster_rcnn_r50_fpn_1x_FLIR_search_r3.py 8
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,8 bash tools/dist_train_search.sh configs/FLIR/faster_rcnn_r50_fpn_1x_FLIR_search_r4.py 8
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,8 bash tools/dist_train_search.sh configs/FLIR/faster_rcnn_r50_fpn_1x_FLIR_search_r5.py 8
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,8 bash tools/dist_train_search.sh configs/FLIR/faster_rcnn_r50_fpn_1x_FLIR_search_r6.py 8
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,8 bash tools/dist_train_search.sh configs/FLIR/faster_rcnn_r50_fpn_1x_FLIR_search_r7.py 8
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,8 bash tools/dist_train_search.sh configs/FLIR/faster_rcnn_r50_fpn_1x_FLIR_search_r8.py 8
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,8 bash tools/dist_train_search.sh configs/FLIR/faster_rcnn_r50_fpn_1x_FLIR_search_r9.py 8
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,8 bash tools/dist_train_search.sh configs/FLIR/faster_rcnn_r50_fpn_1x_FLIR_search_r10.py 8


bash tools/dist_train.sh configs/FLIR_cls/resnet50_ensemble_0p1.py 8
bash tools/dist_train.sh configs/FLIR_cls/resnet50_ensemble_0p01.py 8
bash tools/dist_train.sh configs/FLIR_cls/resnet50_ensemble_0p0001.py 8
bash tools/dist_train.sh configs/FLIR_cls/resnet50_ensemble_0p002.py 8
bash tools/dist_train.sh configs/FLIR_cls/resnet50_ensemble_0p003.py 8
bash tools/dist_train.sh configs/FLIR_cls/resnet50_ensemble_0p05.py 8
bash tools/dist_train.sh configs/FLIR_cls/resnet50_ensemble_0p005.py 8
bash tools/dist_train.sh configs/FLIR_cls/resnet50_ensemble_0p0005.py 8

