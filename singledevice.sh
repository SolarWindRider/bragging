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

nohup python feature_disentangle.py -dv cuda:0 -n p9 -a 0.9 -b 0.9 -c 1.0 -d 0.9 &# 这里的mlpinit是True,运行之后重改了代码
nohup python feature_disentangle.py -dv cuda:1 -n p5 -a 1.0 -b 1.0 -c 1.0 -d 1.0 &
nohup python feature_disentangle.py -dv cuda:2 -n p6 -a 0.9 -b 0.9 -c 1.0 -d 0.9 &

nohup python feature_disentangle.py -dv cuda:0 -n p7 -isClassShuffle False -ismlpinit True -ep 40 -a 1.0 -b 1.0 -c 1.0 -d 1.0 &
nohup python feature_disentangle.py -dv cuda:1 -n p8 -isClassShuffle False -ismlpinit True -ep 80 -a 0.9 -b 0.9 -c 1.0 -d 0.9 &
nohup python feature_disentangle.py -dv cuda:2 -n p10 -isClassShuffle True -ismlpinit False -ep 50 -a 0.9 -b 0.9 -c 1.0 -d 0.9 &

nohup python feature_disentangle.py -dv cuda:0 -n p11 -isClassShuffle False -ismlpinit False -ep 50 -a 1.0 -b 1.0 -c 1.0 -d 1.0 &
nohup python feature_disentangle.py -dv cuda:1 -n p12 -isClassShuffle False -ismlpinit False -ep 50 -a 1.0 -b 1.0 -c 1.1 -d 1.0 &
nohup python feature_disentangle.py -dv cuda:2 -n p13 -isClassShuffle False -ismlpinit False -ep 50 -a 1.0 -b 1.0 -c 1.2 -d 1.0 &

nohup python feature_disentangle.py -dv cuda:0 -n p14 -isClassShuffle True -ismlpinit False -ep 50 -a 1.0 -b 1.0 -c 1.1 -d 1.0 &
nohup python feature_disentangle.py -dv cuda:1 -n p15 -isClassShuffle True -ismlpinit False -ep 50 -a 0.2 -b 0.2 -c 1.0 -d 0.5 &
nohup python feature_disentangle.py -dv cuda:2 -n p16 -isClassShuffle True -ismlpinit False -ep 50 -a 0.2 -b 0.2 -c 1.1 -d 0.5 &

nohup python feature_disentangle.py -dv cuda:0 -n p17 -isClassShuffle True -ismlpinit False -ep 50 -a 1.1 -b 1.1 -c 1.0 -d 1.1 &
nohup python feature_disentangle.py -dv cuda:1 -n p18 -isClassShuffle True -ismlpinit False -ep 50 -a 1.3 -b 1.3 -c 1.0 -d 1.3 &
nohup python feature_disentangle.py -dv cuda:2 -n p19 -isClassShuffle True -ismlpinit False -ep 50 -a 1.5 -b 1.5 -c 1.0 -d 1.5 &

nohup python feature_disentangle.py -dv cuda:0 -n p20 -ep 50 -a 0.9 -b 0.9 -c 1.0 -d 0.9 &
nohup python feature_disentangle.py -dv cuda:1 -n p21 -ep 50 -a 0.2 -b 0.2 -c 1.0 -d 0.5 &
nohup python feature_disentangle.py -dv cuda:2 -n p22 -ep 50 -a 1.3 -b 1.3 -c 1.0 -d 1.3 &

nohup python feature_disentangle.py -dv cuda:0 -n p23 -isClassShuffle False -ep 50 -a 1.3 -b 1.3 -c 1.0 -d 1.3 &
nohup python feature_disentangle.py -dv cuda:1 -n p24 -isClassShuffle True -ep 50 -a 1.5 -b 1.5 -c 1.0 -d 1.5 &
nohup python feature_disentangle.py -dv cuda:2 -n p25 -isClassShuffle False -ep 50 -a 1.5 -b 1.5 -c 1.0 -d 1.5 &

nohup python feature_disentangle.py -dv cuda:0 -n p26 -isClassShuffle False -ep 100 -a 1.3 -b 1.3 -c 1.0 -d 1.3 &
nohup python feature_disentangle.py -dv cuda:1 -n p27 -isClassShuffle False -ep 100 -a 1.5 -b 1.5 -c 1.0 -d 1.5 &
nohup python feature_disentangle.py -dv cuda:2 -n p28 -isClassShuffle False -ep 100 -a 1.7 -b 1.7 -c 1.0 -d 1.7 &
# 从这列开始，鉴别器的dropout变为0.2
nohup python feature_disentangle.py -dv cuda:0 -n p29 -isClassShuffle False -ep 120 -a 1.5 -b 1.5 -c 1.0 -d 1.5 &
nohup python feature_disentangle.py -dv cuda:1 -n p30 -isClassShuffle False -ep 120 -a 1.7 -b 1.7 -c 1.0 -d 1.7 &
nohup python feature_disentangle.py -dv cuda:2 -n p31 -isClassShuffle False -ep 120 -a 1.9 -b 1.9 -c 1.0 -d 1.9 &

# 鉴别器的dropout变回0.5
nohup python feature_disentangle.py -dv cuda:1 -n p32 -clm 2 -isClassShuffle False -ep 50 -a 1.5 -b 1.5 -c 1.0 -d 1.5 &
nohup python ablation.py -dv cuda:0 -n p33 -isClassShuffle False -ep 50 -a 1.5 -b 1.5 -c 1.0 -d 1.5 &
nohup python feature_disentangle.py -dv cuda:2 -n p34 -isClassShuffle False -ep 50 -a 1.5 -b 1.5 -c 1.0 -d 1.5 &

nohup python feature_disentangle.py -dv cuda:0 -n p37 -isClassShuffle False -ep 50 -c 1.0 -d 0.05 &

nohup python feature_disentangle.py -dv cuda:0 -n p38 -clm 2 -isClassShuffle False -ep 50 -a 1.1 -b 1.1 -c 1.0 -d 1.1 &
nohup python feature_disentangle.py -dv cuda:1 -n p39 -clm 2 -isClassShuffle False -ep 50 -a 0.9 -b 0.9 -c 1.0 -d 0.9 &
nohup python feature_disentangle.py -dv cuda:2 -n p40 -clm 2 -isClassShuffle False -ep 50 -a 1.3 -b 1.3 -c 1.0 -d 1.3 &

nohup python feature_disentangle.py -dv cuda:0 -n p41 -clm 2 -isClassShuffle False -ep 50 -a 1.0 -b 1.0 -c 1.0 -d 1.0 &
nohup python feature_disentangle.py -dv cuda:1 -n p42 -clm 2 -isClassShuffle False -ep 50 -a 1.1 -b 1.1 -c 1.0 -d 1.1 &
nohup python feature_disentangle.py -dv cuda:2 -n p43 -clm 2 -isClassShuffle False -ep 50 -a 0.9 -b 0.9 -c 1.0 -d 0.9 &

nohup python feature_disentangle.py -dv cuda:0 -n p44 -clm 2 -d_extra 0.0 -isClassShuffle False -ep 50 -a 1.0 -b 1.0 -c 1.0 &
nohup python feature_disentangle.py -dv cuda:1 -n p45 -clm 2 -d_extra 0.0 -isClassShuffle False -ep 50 -a 0.5 -b 0.5 -c 1.0 &
nohup python feature_disentangle.py -dv cuda:2 -n p46 -clm 2 -d_extra 0.0 -isClassShuffle False -ep 50 -a 1.5 -b 1.5 -c 1.0 &

nohup python feature_disentangle.py -dv cuda:0 -n p47 -clm 2 -d_extra 0.0 -isClassShuffle False -ep 50 -a 1.5 -b 1.5 -c 1.0 -d 1.5 &
nohup python feature_disentangle.py -dv cuda:1 -n p48 -clm 2 -d_extra 0.2 -isClassShuffle False -ep 50 -a 1.5 -b 1.5 -c 1.0 -d 1.0 &
nohup python feature_disentangle.py -dv cuda:2 -n p49 -clm 2 -d_extra 0.2 -isClassShuffle False -ep 50 -a 1.5 -b 1.5 -c 1.0 -d 1.5 &

nohup python feature_disentangle_bin.py -dv cuda:0 -n p50 -clm 2 -d_extra 0.0 -isClassShuffle False -ep 50 -a 1.5 -b 1.5 -c 1.0 &
nohup python feature_disentangle_bin.py -dv cuda:1 -n p51 -clm 2 -d_extra 0.0 -isClassShuffle False -ep 50 -a 1.0 -b 1.0 -c 1.0 &
nohup python feature_disentangle_bin.py -dv cuda:2 -n p52 -clm 2 -d_extra 0.0 -isClassShuffle False -ep 50 -a 0.5 -b 0.5 -c 1.0 &

nohup python feature_disentangle_bin.py -dv cuda:0 -n p53 -clm 2 -d_extra 0.0 -isClassShuffle False -ep 50 -a 1.7 -b 1.7 -c 1.0 &
nohup python feature_disentangle_bin.py -dv cuda:1 -n p54 -clm 2 -d_extra 0.0 -isClassShuffle False -ep 50 -a 1.9 -b 1.9 -c 1.0 &
nohup python feature_disentangle_bin.py -dv cuda:2 -n p55 -clm 2 -d_extra 0.0 -isClassShuffle False -ep 50 -a 2.1 -b 2.1 -c 1.0 &

nohup python feature_disentangle.py -dv cuda:0 -n p56 -clm 2 -d_extra 0.0 -isClassShuffle False -ep 50 -a 1.7 -b 1.7 -c 1.0 -d 0.005 &
nohup python feature_disentangle.py -dv cuda:1 -n p57 -clm 2 -d_extra 0.0 -isClassShuffle False -ep 50 -a 1.7 -b 1.7 -c 1.0 -d 0.003 &
nohup python feature_disentangle.py -dv cuda:2 -n p58 -clm 2 -d_extra 0.0 -isClassShuffle False -ep 50 -a 1.7 -b 1.7 -c 1.0 -d 0.001 &

nohup python feature_disentangle.py -dv cuda:0 -n p59 -clm 2 -d_extra 0.0 -isClassShuffle False -ep 50 -a 1.7 -b 1.7 -c 1.0 -d 0.025 &
nohup python feature_disentangle.py -dv cuda:1 -n p60 -clm 2 -d_extra 0.0 -isClassShuffle False -ep 50 -a 1.7 -b 1.7 -c 1.0 -d 0.035 &
nohup python feature_disentangle.py -dv cuda:2 -n p61 -clm 2 -d_extra 0.0 -isClassShuffle False -ep 50 -a 1.7 -b 1.7 -c 1.0 -d 0.004 &

nohup python ablation.py -dv cuda:0 -n p62 -isClassShuffle False -ep 100 -c 1.0 -d 1.0 &
nohup python ablation.py -dv cuda:1 -n p63 -isClassShuffle False -ep 100 -c 1.0 -d 0.5 &
nohup python ablation.py -dv cuda:2 -n p64 -isClassShuffle False -ep 100 -c 1.0 -d 1.5 &

nohup python ablation.py -dv cuda:0 -n p65 -isClassShuffle False -ep 100 -c 1.0 -d 1.7 &
nohup python ablation.py -dv cuda:1 -n p66 -isClassShuffle False -ep 100 -c 1.0 -d 1.9 &
nohup python ablation.py -dv cuda:2 -n p67 -isClassShuffle False -ep 100 -c 1.0 -d 2.1 &

nohup python ablation.py -dv cuda:0 -n p68 -isClassShuffle False -ep 100 -c 1.0 -d 0.1 &
nohup python ablation.py -dv cuda:1 -n p69 -isClassShuffle False -ep 100 -c 1.0 -d 0.3 &
nohup python ablation.py -dv cuda:2 -n p70 -isClassShuffle False -ep 100 -c 1.0 -d 0.7 &

nohup python ablation.py -dv cuda:0 -n p71 -isClassShuffle False -ep 100 -c 1.0 -d 1.4 &
nohup python ablation.py -dv cuda:1 -n p72 -isClassShuffle False -ep 100 -c 1.0 -d 1.45 &
nohup python ablation.py -dv cuda:2 -n p73 -isClassShuffle False -ep 100 -c 1.0 -d 1.55 &

