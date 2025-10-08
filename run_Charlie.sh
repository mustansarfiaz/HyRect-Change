#Initial settings
cd /dccstor/urban/mustansar/codes/cd/bi_directional4/
conda activate /dccstor/urban/mustansar/envs/cd_cf
export PYTHONPATH=$PYTHONPATH:$(pwd)


CUDA_VISIBLE_DEVICES=0,1 python main_cd.py  --batch_size 64


CUDA_VISIBLE_DEVICES=0,1 python main_cd_infer.py 