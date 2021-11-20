#!/bin/sh

target_size=$1
CS=5
cur_epochs=1000
source /home/efyang/projects/nerf_atlas/env/bin/activate

make clean
python3 -O runner.py -d data/lego_ortho/ --data-kind ortho \
    --size $CS --epochs $cur_epochs --save models/lego_ortho_progressive.pt \
    --batch-size 2	--model fvr -lr 1e-3 --save-freq=500 --clip-gradients 1e-2 \
    --loss-fns l2 --valid-freq 100 --refl-kind fvrview --sigmoid-kind leaky_relu --notest --notraintest
NOW=$( date '+%F_%H-%M-%S' )
dir="$CS""_$NOW"
mkdir outputs/$dir
mv outputs/training_loss.png outputs/$dir/training_loss.png
cat outputs/valid_*.png | ffmpeg -f image2pipe -r 15 -i - outputs/$dir/valid_image.mp4

while [ $CS -le $target_size ]
do
    ((CS=CS+8))
    ((cur_epochs=cur_epochs+1000))
    make clean
    echo "train $CS for $cur_epochs epochs"
    python3 -O runner.py -d data/lego_ortho/ --data-kind ortho \
        --size $CS --epochs $cur_epochs --save models/lego_ortho_progressive.pt \
        --batch-size 2	--model fvr -lr 1e-3 --save-freq=500 --clip-gradients 1e-2 \
        --loss-fns l2 --valid-freq 100 --refl-kind fvrview --sigmoid-kind leaky_relu  --load models/lego_ortho_progressive.pt --notest --notraintest
    NOW=$( date '+%F_%H-%M-%S' )
    dir="$CS""_$NOW"
    mkdir outputs/$dir
    mv outputs/training_loss.png outputs/$dir/training_loss.png
    cat outputs/valid_*.png | ffmpeg -f image2pipe -r 15 -i - outputs/$dir/valid_image.mp4
done

((cur_epochs=cur_epochs*2))
make clean
echo "train $target_size for $cur_epochs epochs"
python3 -O runner.py -d data/lego_ortho/ --data-kind ortho \
    --size $target_size --epochs $cur_epochs --save models/lego_ortho_progressive.pt \
    --batch-size 2	--model fvr -lr 1e-3 --save-freq=500 --clip-gradients 1e-2 \
    --loss-fns l2 --valid-freq 100 --refl-kind fvrview --sigmoid-kind leaky_relu  --load models/lego_ortho_progressive.pt
NOW=$( date '+%F_%H-%M-%S' )
dir="$CS""_$NOW"
cat outputs/valid_*.png | ffmpeg -f image2pipe -r 15 -i - outputs/valid_image_$dir.mp4
cat outputs/test*.png | ffmpeg -f image2pipe -r 15 -i - outputs/test_$dir.mp4
cat outputs/train*.png | ffmpeg -f image2pipe -r 15 -i - outputs/train_$dir.mp4