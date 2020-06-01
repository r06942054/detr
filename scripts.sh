# 550 Server

cd cd renjie/GitHub/
git clone https://github.com/facebookresearch/detr.git

conda create --name detr python=3.7
conda install -c pytorch pytorch torchvision
conda install cython scipy
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'



cd
mkdir coco
cd coco

wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/zips/test2017.zip

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/image_info_test2017.zip

unzip train2017.zip
unzip val2017.zip
unzip test2017.zip

unzip annotations_trainval2017.zip
unzip image_info_test2017.zip


cd /home/omnieyes/renjie/GitHub/detr/
conda activate detr

CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --coco_path /home/omnieyes/coco
python main.py --batch_size 2 --no_aux_loss --eval --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --coco_path /home/omnieyes/coco

CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --coco_path /home/omnieyes/renjie/GitHub/CenterNet/data/omnieyes --dataset_file omni --num_queries 30 --batch_size 4 --output_dir /home/omnieyes/renjie/GitHub/detr/resnet18_raw/ --backbone resnet18