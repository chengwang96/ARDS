CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model vit3d --batch_size 16 --use_smote --modal mediastinum_window
CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model vit3d --batch_size 16 --use_smote --modal lung_window
CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model vit3d --batch_size 16 --use_smote --modal abdomen

CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model vit3d --batch_size 16 --use_smote --modal mediastinum_window --rotate
CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model vit3d --batch_size 16 --use_smote --modal lung_window --rotate
CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model vit3d --batch_size 16 --use_smote --modal abdomen --rotate

CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model vit3d --batch_size 16 --use_smote --modal mediastinum_window --flip
CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model vit3d --batch_size 16 --use_smote --modal lung_window --flip
CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model vit3d --batch_size 16 --use_smote --modal abdomen --flip

CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model vit3d --batch_size 16 --use_smote --modal mediastinum_window --intensity
CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model vit3d --batch_size 16 --use_smote --modal lung_window --intensity
CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model vit3d --batch_size 16 --use_smote --modal abdomen --intensity

CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model vit3d --batch_size 16 --use_smote --modal mediastinum_window --rotate --flip --intensity
CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model vit3d --batch_size 16 --use_smote --modal lung_window --rotate --flip --intensity
CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model vit3d --batch_size 16 --use_smote --modal abdomen --rotate --flip --intensity



CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model med_clip --batch_size 4 --use_smote --modal mediastinum_window
CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model med_clip --batch_size 4 --use_smote --modal lung_window
CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model med_clip --batch_size 4 --use_smote --modal abdomen

CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model med_clip --batch_size 4 --use_smote --modal mediastinum_window --rotate
CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model med_clip --batch_size 4 --use_smote --modal lung_window --rotate
CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model med_clip --batch_size 4 --use_smote --modal abdomen --rotate

CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model med_clip --batch_size 4 --use_smote --modal mediastinum_window --flip
CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model med_clip --batch_size 4 --use_smote --modal lung_window --flip
CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model med_clip --batch_size 4 --use_smote --modal abdomen --flip

CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model med_clip --batch_size 4 --use_smote --modal mediastinum_window --intensity
CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model med_clip --batch_size 4 --use_smote --modal lung_window --intensity
CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model med_clip --batch_size 4 --use_smote --modal abdomen --intensity

CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model med_clip --batch_size 4 --use_smote --modal mediastinum_window --rotate --flip --intensity
CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model med_clip --batch_size 4 --use_smote --modal lung_window --rotate --flip --intensity
CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model med_clip --batch_size 4 --use_smote --modal abdomen --rotate --flip --intensity
