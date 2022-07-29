nohup python main.py -dv cuda:0 -lm vinai/bertweet-base -n gan_clfonly -c 7 -fp ./models/gan/ -ns 0.5 -sd 3407 &
nohup python main.py -dv cuda:1 -lm vinai/bertweet-base -n gan_clfonly -c 7 -fp ./models/gan/ -ns 0.4 -sd 3407 &

