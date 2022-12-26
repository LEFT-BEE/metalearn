#!/bin/bash

cu_num=1
data=cifar_fs


for p in    1 ; do
for t in  1 5 10; do
for m in 0.5 1 5 10; do
for n in 0.5 1 5 10; do 

name=mag_${m}_mag2_${n}_KL_caseConcent_not_${data}_5shot_${p}_temp_${t}

python ./Cossimg_layermag_concent_not.py --folder=~/data \
		         --dataset=${data} \
		         --model=4conv_sep_not \
		         --device=cuda:${cu_num} \
		         --num-ways=5 \
		         --num-shots=5 \
		         --extractor-step-size=0.5 \
		         --classifier-step-size=0.0 \
		         --fixed-classifier-step-size=0 \
		         --fixed-last-step-size=0 \
		         --meta-lr=1e-3 \
                         --batch-iter=300 \
		         --download\
		         --save-name=${name} \
                         --prefix_file=${name} \
                         --temp=${t} \
				 --output_folder=./output_not\
		         --mag=${m} \
				 --mag2=${n}


# python ./test_plot_revised.py --folder=~/data \
#                  --dataset=${data} \
#                  --model=4conv_sep_not \
#                  --device=cuda:${cu_num} \
#                  --num-ways=5 \
#                  --num-shots=5 \
#                  --extractor-step-size=0.5 \
#                  --classifier-step-size=0.0 \
# 		 --fixed-classifier-step-size=0 \
# 		 --fixed-last-step-size=0 \
#                  --meta-lr=1e-3 \
#                  --download \
#                  --save-name=${name} \
# 				 --output_folder=./output_not\
#                  --temp=${t} \
# 		 --mag=${m} \
# 		 --mag2=${n}

done
done
done
done

echo "finished"