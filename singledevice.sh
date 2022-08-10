nohup python feature_disentangle.py -dv cuda:0 -lm vinai/bertweet-base -n fd1 -fp ./models/gan/ -sd 3407 -h_rep False -loss2bert True &

nohup python feature_disentangle.py -dv cuda:0 -lm vinai/bertweet-base -n fd4 -fp ./models/gan/ -sd 3407 -h_rep True -loss2bert False &
nohup python feature_disentangle.py -dv cuda:1 -lm vinai/bertweet-base -n fd6 -fp ./models/gan/ -sd 3407 -h_rep False -loss2bert False &
nohup python feature_disentangle.py -dv cuda:2 -lm vinai/bertweet-base -n fd5 -fp ./models/gan/ -sd 3407 -h_rep True -loss2bert False &
nohup python feature_disentangle.py -dv cuda:0 -lm vinai/bertweet-base -n fd7 -fp ./models/gan/ -sd 3407 -h_rep False -loss2bert False &

nohup python feature_disentangle.py -dv cuda:1 -lm vinai/bertweet-base -n fd9 -fp ./models/gan/ -sd 3407 -h_rep False -loss2bert True -d_lr 3e-4 -g_lr 3e-6 &
nohup python feature_disentangle.py -dv cuda:2 -lm vinai/bertweet-base -n fd8 -fp ./models/gan/ -sd 3407 -h_rep False -loss2bert True -d_lr 3e-4 -g_lr 3e-6 &

nohup python feature_disentangle.py -dv cuda:0 -lm vinai/bertweet-base -n fd10 -fp ./models/gan/ -sd 3407 -h_rep False -loss2bert True -d_lr 3e-6 -g_lr 3e-6 &
nohup python feature_disentangle.py -dv cuda:1 -lm vinai/bertweet-base -n fd11 -fp ./models/gan/ -sd 3407 -h_rep False -loss2bert True -d_lr 3e-6 -g_lr 3e-6 &
nohup python feature_disentangle.py -dv cuda:2 -lm vinai/bertweet-base -n fd12 -fp ./models/gan/ -sd 3407 -h_rep False -loss2bert True -d_lr 3e-4 -g_lr 3e-6 &

nohup python feature_disentangle.py -dv cuda:0 -lm vinai/bertweet-base -n fd14 -fp ./models/gan/ -sd 3407 -h_rep False -loss2bert True -d_lr 3e-4 -g_lr 3e-6 &
nohup python feature_disentangle.py -dv cuda:1 -lm vinai/bertweet-base -n fd15 -fp ./models/gan/ -sd 3407 -h_rep False -loss2bert True -d_lr 3e-4 -g_lr 3e-6 &

nohup python feature_disentangle.py -dv cuda:0 -lm vinai/bertweet-base -n fd16 -fp ./models/gan/ -sd 3407 -h_rep False -loss2bert True -d_lr 3e-4 -g_lr 3e-6 &
nohup python feature_disentangle.py -dv cuda:1 -lm vinai/bertweet-base -n fd18 -fp ./models/gan/ -sd 3407 -h_rep False -loss2bert True -d_lr 3e-4 -g_lr 3e-6 &
nohup python feature_disentangle.py -dv cuda:2 -lm vinai/bertweet-base -n fd17 -fp ./models/gan/ -sd 3407 -h_rep False -loss2bert True -d_lr 3e-4 -g_lr 3e-6 &

nohup python feature_disentangle.py -dv cuda:0 -n fd19 -h_rep False -loss2bert True -d_lr 3e-4 -g_lr 3e-6 -d_extra 0.2 &
nohup python feature_disentangle.py -dv cuda:1 -n fd20 -h_rep False -loss2bert True -d_lr 3e-4 -g_lr 3e-6 -d_extra 0.2 &
nohup python feature_disentangle.py -dv cuda:2 -n fd21 -h_rep True -loss2bert True -d_lr 3e-4 -g_lr 3e-6 -d_extra 0.2 &

nohup python feature_disentangle.py -dv cuda:0 -n f22 -h_rep True -loss2bert True -d_lr 3e-4 -g_lr 3e-6 -d_extra 0.2 &
nohup python feature_disentangle.py -dv cuda:1 -n f23 -h_rep True -loss2bert True -d_lr 3e-4 -g_lr 3e-6 -d_extra 0.2 -fst_d_extra 0.6 &

nohup python feature_disentangle.py -dv cuda:2 -n f24 -h_rep True -loss2bert True -d_lr 3e-4 -g_lr 3e-6 -d_extra 0.2 -fst_d_extra 0.6 &
nohup python feature_disentangle.py -dv cuda:0 -n f25 -h_rep True -loss2bert True -d_lr 3e-4 -g_lr 3e-6 -d_extra 0.2 -fst_d_extra 0.6 &
nohup python feature_disentangle.py -dv cuda:1 -n f26 -h_rep False -loss2bert True -d_lr 3e-4 -g_lr 3e-6 -d_extra 0.2 -fst_d_extra 0.6 &
