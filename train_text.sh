CUDA_VISIBLE_DEVICES=0,1 python propriety_text_rnn.py --gpu=2 --epochs=100 --batch_size=32  --data_path='/path/dataset/text/' --snapshot='' --model_path='./model/textmodel/'
