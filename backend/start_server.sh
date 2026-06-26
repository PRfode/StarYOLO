export CUDA_VISIBLE_DEVICES=0

python ./backend/server.py \
    --weights ./res/model/best.pt \
    --model_yaml ultralytics/cfg/models/staryolon.yaml \
    --device 0