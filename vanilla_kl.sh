#!/bin/bash

cu_num=0
data=miniimagenet



for p in    1      ; do
for t in  1 2 3 4; do
for m in 1 0.5; do

name=mag_${m}_KL_case1_${data}_5shot_${p}_temp_${t}


                
python ./Cossimg_layermag_kl.py --folder=~/data \
		         --dataset=${data} \
		         --model=4conv_sep \
		         --device=cuda:${cu_num} \
		         --num-ways=5 \
		         --num-shots=5 \
		         --extractor-step-size=0.5 \
		         --classifier-step-size=0.0 \
		         --fixed-classifier-step-size=0 \
		         --fixed-last-step-size=0 \
		         --meta-lr=1e-3 \
                         --batch-iter=300 \
		         --download \
		         --save-name=${name} \
                         --prefix_file=${name} \
                         --temp=${t} \
		         --mag=${m}


python ./test_plot.py --folder=~/data \
                 --dataset=${data} \
                 --model=4conv_sep \
                 --device=cuda:${cu_num} \
                 --num-ways=5 \
                 --num-shots=5 \
                 --extractor-step-size=0.5 \
                 --classifier-step-size=0.0 \
		 --fixed-classifier-step-size=0 \
		 --fixed-last-step-size=0 \
                 --meta-lr=1e-3 \
                 --download \
                 --save-name=${name} \
                 --output-folder=./output \
                 --temp=${t} \
		 --mag=${m}



done
done
done



for p in    1      ; do
for t in  1 2 3 4; do
for m in 1 0.5; do

name=mag_${m}_KL_case2_${data}_5shot_${p}_temp_${t}


                
python ./Cossimg_layermag_kl_2.py --folder=~/data \
		         --dataset=${data} \
		         --model=4conv_sep \
		         --device=cuda:${cu_num} \
		         --num-ways=5 \
		         --num-shots=5 \
		         --extractor-step-size=0.5 \
		         --classifier-step-size=0.0 \
		         --fixed-classifier-step-size=0 \
		         --fixed-last-step-size=0 \
		         --meta-lr=1e-3 \
                         --batch-iter=300 \
		         --download \
		         --save-name=${name} \
                         --prefix_file=${name} \
                         --temp=${t} \
		         --mag=${m}


python ./test_plot.py --folder=~/data \
                 --dataset=${data} \
                 --model=4conv_sep \
                 --device=cuda:${cu_num} \
                 --num-ways=5 \
                 --num-shots=5 \
                 --extractor-step-size=0.5 \
                 --classifier-step-size=0.0 \
		 --fixed-classifier-step-size=0 \
		 --fixed-last-step-size=0 \
                 --meta-lr=1e-3 \
                 --download \
                 --save-name=${name} \
                 --output-folder=./output \
                 --temp=${t} \
		 --mag=${m}



done
done
done










for p in    1      ; do
for t in  1 2 3 4; do
for m in 1 0.5; do

name=mag_${m}_KL_case3_${data}_5shot_${p}_temp_${t}


                
python ./Cossimg_layermag_kl_3.py --folder=~/data \
		         --dataset=${data} \
		         --model=4conv_sep \
		         --device=cuda:${cu_num} \
		         --num-ways=5 \
		         --num-shots=5 \
		         --extractor-step-size=0.5 \
		         --classifier-step-size=0.0 \
		         --fixed-classifier-step-size=0 \
		         --fixed-last-step-size=0 \
		         --meta-lr=1e-3 \
                         --batch-iter=300 \
		         --download \
		         --save-name=${name} \
                         --prefix_file=${name} \
                         --temp=${t} \
		         --mag=${m}


python ./test_plot.py --folder=~/data \
                 --dataset=${data} \
                 --model=4conv_sep \
                 --device=cuda:${cu_num} \
                 --num-ways=5 \
                 --num-shots=5 \
                 --extractor-step-size=0.5 \
                 --classifier-step-size=0.0 \
		 --fixed-classifier-step-size=0 \
		 --fixed-last-step-size=0 \
                 --meta-lr=1e-3 \
                 --download \
                 --save-name=${name} \
                 --output-folder=./output \
                 --temp=${t} \
		 --mag=${m}



done
done
done





























echo "finished"
