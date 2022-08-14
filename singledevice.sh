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

nohup python feature_disentangle.py -dv cuda:0 -n f27 -h_rep True -loss2bert True -fst_d_extra 0.8 -d_extra 0.4 -d_lr 3e-3 -g_lr 3e-6 &
nohup python feature_disentangle.py -dv cuda:1 -n f28 -h_rep True -loss2bert True -fst_d_extra 1.0 -d_extra 0.4 -d_lr 3e-3 -g_lr 3e-6 &
nohup python feature_disentangle.py -dv cuda:2 -n f29 -h_rep True -loss2bert True -fst_d_extra 1.0 -d_extra 0.4 -d_lr 3e-2 -g_lr 3e-6 &
nohup python feature_disentangle.py -dv cuda:2 -n f30 -h_rep True -loss2bert True -fst_d_extra 0.8 -d_extra 0.4 -d_lr 1e-3 -g_lr 3e-6 &

nohup python feature_disentangle.py -dv cuda:0 -n f31 -h_rep True -loss2bert True -fst_d_extra 0.6 -d_extra 0.2 -d_clip True -d_lr 3e-4 -g_lr 3e-6 &
nohup python feature_disentangle.py -dv cuda:1 -n f32 -h_rep True -loss2bert True -fst_d_extra 0.6 -d_extra 0.2 -d_clip False -d_lr 5e-4 -g_lr 3e-6 &
nohup python feature_disentangle.py -dv cuda:2 -n f33 -h_rep True -loss2bert True -fst_d_extra 0.6 -d_extra 0.2 -d_clip False -d_lr 3e-4 -g_lr 3e-6 &

nohup python feature_disentangle.py -dv cuda:2 -n f34 -h_rep True -loss2bert True -fst_d_extra 0.6 -d_extra 0.2 -d_clip False -d_lr 3e-4 -g_lr 3e-6 &
nohup python feature_disentangle.py -dv cuda:2 -n f35 -h_rep True -loss2bert True -fst_d_extra 0.6 -d_extra 0.2 -d_clip False -d_lr 3e-4 -g_lr 3e-6 &

nohup python feature_disentangle.py -dv cuda:0 -n p1 -a 1.0 -b 1.0 -c 1.0 -d 1.0 &
nohup python feature_disentangle.py -dv cuda:1 -n p2 -a 0.8 -b 0.8 -c 1.0 -d 0.8 &
nohup python feature_disentangle.py -dv cuda:2 -n p3 -a 0.6 -b 0.6 -c 1.0 -d 0.6 &
nohup python feature_disentangle.py -dv cuda:2 -n p4 -a 0.7 -b 0.7 -c 1.0 -d 0.7 &

nohup python feature_disentangle.py -dv cuda:0 -n p9 -a 0.9 -b 0.9 -c 1.0 -d 0.9 & # 这里的mlpinit是True,运行之后重改了代码
nohup python feature_disentangle.py -dv cuda:1 -n p5 -a 1.0 -b 1.0 -c 1.0 -d 1.0 &
nohup python feature_disentangle.py -dv cuda:2 -n p6 -a 0.9 -b 0.9 -c 1.0 -d 0.9 &

nohup python feature_disentangle.py -dv cuda:0 -n p7  -isClassShuffle False -ismlpinit True  -ep 40 -a 1.0 -b 1.0 -c 1.0 -d 1.0 &
nohup python feature_disentangle.py -dv cuda:1 -n p8  -isClassShuffle False -ismlpinit True  -ep 80 -a 0.9 -b 0.9 -c 1.0 -d 0.9 &
nohup python feature_disentangle.py -dv cuda:2 -n p10 -isClassShuffle True  -ismlpinit False -ep 50 -a 0.9 -b 0.9 -c 1.0 -d 0.9 &

