export CUDA_VISIBLE_DEVICES=0

python ./backend/server.py \
    --weights ./res/model/yolov8n.pt \
    --model_yaml ultralytics/cfg/models/yolov8n.yaml \
    --device 0