#!/bin/bash

scp -r $myOVH:~/nn-visu/Figures ./results/
scp -r $myOVH:~/nn-visu/slides+report/ ./results/
exit
