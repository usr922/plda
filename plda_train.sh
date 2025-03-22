SESSION_NAME=plda

Train_cam=work_dir/${SESSION_NAME}/train_cam
Trainaug_cam=work_dir/${SESSION_NAME}/trainaug_cam
Train_crf=work_dir/${SESSION_NAME}/train_crf
Val_cam=work_dir/${SESSION_NAME}/val_cam
Val_crf=work_dir/${SESSION_NAME}/val_crf

mkdir work_dir/${SESSION_NAME}
mkdir ${Train_cam}
mkdir ${Train_crf}
mkdir ${Val_cam}
mkdir ${Val_crf}


GPU=


# 1. train plda
CUDA_VISIBLE_DEVICES=${GPU} python plda_train.py \
    --network models.resnet38_plda \
    --weights weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params \
    --session_name ${SESSION_NAME} \
    --batch_size 8


echo 'make cam trainaug....'
# 2. infer cam of train+aug, w/o crf
CUDA_VISIBLE_DEVICES=${GPU} python plda_infer.py \
    --network models.resnet38_plda \
    --weights ${SESSION_NAME}.pth \
    --infer_list voc12/train_aug.txt \
    --out_cam ${Trainaug_cam}


# 3. IRN step
# simply follow https://github.com/jiwoon-ahn/irn

# 4. Segmentation training
# simply follow https://github.com/usr922/wseg







