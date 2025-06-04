# Building Object Relation via Historical Trajectory for Visual Navigation

### pre-tain
python main_pretrain.py --gpu-ids 0 --workers 4 --model BaseModel --detr --title Pretrain_Model --work-dir ./pretrain_dirs/


# 使用轨迹数据预训练的模型训练最终模型
python main.py --gpu-ids 0 1  --workers 14 --model Model --detr --title Fomal_Train_Model --work-dir ./work_dirs/ --test-after-train --pretrained-trans /pretrain_dirs/pre_train_2022-07-26_16-19-36/trained_models/checkpoint0099.pth --ep-save-freq 100000



# 使用无预训练的模型
python main.py --gpu-ids 0 1  --workers 14 --model Model --detr --title Without_Pretrain_Model --work-dir ./work_dirs/ --test-after-train --ep-save-freq 100000

