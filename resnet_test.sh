#!/bin/bash

nb=20
#i=2
#while [ $i -le $nb ]
#do
#	echo "python2.7 final_pipeline.py --tmodel resnet50 --tmethod bow --ntry 1 --start "$i
#	python2.7 final_pipeline.py --tmodel resnet50 --tmethod bow --ntry 1 --start $i
#	i=$((i+1))
#done
#j=1
#while [ $j -le $nb ]
#do
#	echo "python2.7 final_pipeline.py --tmodel resnet50 --tmethod sift --ntry 1 --start "$j
#	python2.7 final_pipeline.py --tmodel resnet50 --tmethod sift --ntry 1 --start $j
#	j=$((j+1))
#done
#k=1
k=2
while [ $k -le $nb ]
do
	echo "python2.7 final_pipeline.py --tmodel resnet50 --tmethod harris --ntry 1 --start "$k
	python2.7 final_pipeline.py --tmodel resnet50 --tmethod harris --ntry 1 --start $k
	k=$((k+1))
done

exit
