CUDA_VISIBLE_DEVICES=0 python ards_cls.py --model vit3d --batch_size 4 --modal mediastinum_window
CUDA_VISIBLE_DEVICES=1 python ards_cls.py --model vit3d --batch_size 4 --modal lung_window
CUDA_VISIBLE_DEVICES=2 python ards_cls.py --model vit3d --batch_size 4 --modal abdomen

CUDA_VISIBLE_DEVICES= python ards_cls.py --model vit3d --batch_size 4 --modal mediastinum_window --rotate
CUDA_VISIBLE_DEVICES= python ards_cls.py --model vit3d --batch_size 4 --modal lung_window --rotate
CUDA_VISIBLE_DEVICES= python ards_cls.py --model vit3d --batch_size 4 --modal abdomen --rotate

CUDA_VISIBLE_DEVICES= python ards_cls.py --model vit3d --batch_size 4 --modal mediastinum_window --flip
CUDA_VISIBLE_DEVICES= python ards_cls.py --model vit3d --batch_size 4 --modal lung_window --flip
CUDA_VISIBLE_DEVICES= python ards_cls.py --model vit3d --batch_size 4 --modal abdomen --flip

CUDA_VISIBLE_DEVICES= python ards_cls.py --model vit3d --batch_size 4 --modal mediastinum_window --intensity
CUDA_VISIBLE_DEVICES= python ards_cls.py --model vit3d --batch_size 4 --modal lung_window --intensity
CUDA_VISIBLE_DEVICES= python ards_cls.py --model vit3d --batch_size 4 --modal abdomen --intensity

CUDA_VISIBLE_DEVICES= python ards_cls.py --model vit3d --batch_size 4 --modal mediastinum_window --rotate --flip --intensity
CUDA_VISIBLE_DEVICES= python ards_cls.py --model vit3d --batch_size 4 --modal lung_window --rotate --flip --intensity
CUDA_VISIBLE_DEVICES= python ards_cls.py --model vit3d --batch_size 4 --modal abdomen --rotate --flip --intensity



CUDA_VISIBLE_DEVICES= python ards_cls.py --model med_clip --batch_size 4 --modal mediastinum_window
CUDA_VISIBLE_DEVICES= python ards_cls.py --model med_clip --batch_size 4 --modal lung_window
CUDA_VISIBLE_DEVICES= python ards_cls.py --model med_clip --batch_size 4 --modal abdomen

CUDA_VISIBLE_DEVICES= python ards_cls.py --model med_clip --batch_size 4 --modal mediastinum_window --rotate
CUDA_VISIBLE_DEVICES= python ards_cls.py --model med_clip --batch_size 4 --modal lung_window --rotate
CUDA_VISIBLE_DEVICES= python ards_cls.py --model med_clip --batch_size 4 --modal abdomen --rotate

CUDA_VISIBLE_DEVICES= python ards_cls.py --model med_clip --batch_size 4 --modal mediastinum_window --flip
CUDA_VISIBLE_DEVICES= python ards_cls.py --model med_clip --batch_size 4 --modal lung_window --flip
CUDA_VISIBLE_DEVICES= python ards_cls.py --model med_clip --batch_size 4 --modal abdomen --flip

CUDA_VISIBLE_DEVICES= python ards_cls.py --model med_clip --batch_size 4 --modal mediastinum_window --intensity
CUDA_VISIBLE_DEVICES= python ards_cls.py --model med_clip --batch_size 4 --modal lung_window --intensity
CUDA_VISIBLE_DEVICES= python ards_cls.py --model med_clip --batch_size 4 --modal abdomen --intensity

CUDA_VISIBLE_DEVICES= python ards_cls.py --model med_clip --batch_size 4 --modal mediastinum_window --rotate --flip --intensity
CUDA_VISIBLE_DEVICES= python ards_cls.py --model med_clip --batch_size 4 --modal lung_window --rotate --flip --intensity
CUDA_VISIBLE_DEVICES= python ards_cls.py --model med_clip --batch_size 4 --modal abdomen --rotate --flip --intensity
