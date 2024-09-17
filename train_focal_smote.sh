CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model vit3d --batch_size 16 --use_smote --modal mediastinum_window --loss focal
CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model vit3d --batch_size 16 --use_smote --modal lung_window --loss focal
CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model vit3d --batch_size 16 --use_smote --modal abdomen --loss focal

CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model vit3d --batch_size 16 --use_smote --modal mediastinum_window --rotate --loss focal
CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model vit3d --batch_size 16 --use_smote --modal lung_window --rotate --loss focal
CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model vit3d --batch_size 16 --use_smote --modal abdomen --rotate --loss focal

CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model vit3d --batch_size 16 --use_smote --modal mediastinum_window --flip --loss focal
CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model vit3d --batch_size 16 --use_smote --modal lung_window --flip --loss focal
CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model vit3d --batch_size 16 --use_smote --modal abdomen --flip --loss focal

CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model vit3d --batch_size 16 --use_smote --modal mediastinum_window --intensity --loss focal
CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model vit3d --batch_size 16 --use_smote --modal lung_window --intensity --loss focal
CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model vit3d --batch_size 16 --use_smote --modal abdomen --intensity --loss focal

CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model vit3d --batch_size 16 --use_smote --modal mediastinum_window --rotate --flip --intensity --loss focal
CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model vit3d --batch_size 16 --use_smote --modal lung_window --rotate --flip --intensity --loss focal
CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model vit3d --batch_size 16 --use_smote --modal abdomen --rotate --flip --intensity --loss focal



CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model med_clip --batch_size 4 --use_smote --modal mediastinum_window --loss focal
CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model med_clip --batch_size 4 --use_smote --modal lung_window --loss focal
CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model med_clip --batch_size 4 --use_smote --modal abdomen --loss focal

CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model med_clip --batch_size 4 --use_smote --modal mediastinum_window --rotate --loss focal
CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model med_clip --batch_size 4 --use_smote --modal lung_window --rotate --loss focal
CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model med_clip --batch_size 4 --use_smote --modal abdomen --rotate --loss focal

CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model med_clip --batch_size 4 --use_smote --modal mediastinum_window --flip --loss focal
CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model med_clip --batch_size 4 --use_smote --modal lung_window --flip --loss focal
CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model med_clip --batch_size 4 --use_smote --modal abdomen --flip --loss focal

CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model med_clip --batch_size 4 --use_smote --modal mediastinum_window --intensity --loss focal
CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model med_clip --batch_size 4 --use_smote --modal lung_window --intensity --loss focal
CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model med_clip --batch_size 4 --use_smote --modal abdomen --intensity --loss focal

CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model med_clip --batch_size 4 --use_smote --modal mediastinum_window --rotate --flip --intensity --loss focal
CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model med_clip --batch_size 4 --use_smote --modal lung_window --rotate --flip --intensity --loss focal
CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model med_clip --batch_size 4 --use_smote --modal abdomen --rotate --flip --intensity --loss focal
