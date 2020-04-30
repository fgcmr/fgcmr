CUDA_VISIBLE_DEVICES=0,1 python main_propriety.py --workers=4 --gpu=2 --epochs=60 --batch_size=64  --data_path='/path/dataset/audio' --snapshot='' --model_path='./model/audiomodel/'
