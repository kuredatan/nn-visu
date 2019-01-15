# coding:utf-8
import sys
sys.path += ['layers', 'models', 'utils']
import argparse
from comp_fmaps import bow_comparison, corresp_comparison, repeatability_harris, load_input

## python3.6 analysis_fmaps.py --tname conv/convfeature_map_layer_conv1.png --tmethod bow

parser = argparse.ArgumentParser(description='Analysis of deconvoluted images')
parser.add_argument('--tname', type=str, default='', metavar='N',
                    help='Name of image to analyze.')
parser.add_argument('--tmethod', type=str, default='bow', metavar='M',
                    help='Method: [\'bow\', \'sift\', \'harris\'].')
args = parser.parse_args()

## Use the CATS dataset
list_img = ["./data/cats/cat"+str(i)+".jpg" for i in range(1, 12)]
assert len(list_img) > 0, "Put some images in the ./data/cats folder"
images_list = [load_input(im_name) for im_name in list_img]
fmap = load_input("./Figures/"+args.tname)

if (args.tmethod == "bow"):
	bow_comparison(fmap, images_list, name="cats", num_words=10, fmap_name="1", list_img=list_img)
if (args.tmethod == "sift"):
	corresp_comparison(fmap, images_list, name="cats", fmap_name="1", list_img=list_img)
if (args.tmethod == "harris"):
	repeatability_harris(fmap, images_list, name="cats", fmap_name="1", list_img=list_img)
