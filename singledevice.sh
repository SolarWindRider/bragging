nohup python main.py -dv cuda:0 -lm bert-base-cased -n bert -c 7 -fp ./models/banlanced/ &
nohup python main.py -dv cuda:1 -lm roberta-base -n roberta -c 7 -fp ./models/banlanced/ &
nohup python main.py -dv cuda:2 -lm vinai/bertweet-base -n bertweet -c 7 -fp ./models/banlanced/ &

