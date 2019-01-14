# coding:utf-8
import sys
sys.path += ['layers', 'models', 'utils']
import argparse
from print_norm_utils import load_input
from comp_fmaps import bow_comparison, corresp_comparison, repeatability_harris

parser = argparse.ArgumentParser(description='Analysis of deconvoluted images')
parser.add_argument('--tname', type=str, default='', metavar='N',
                    help='Name of image to analyze.')
parser.add_argument('--tmethod', type=str, default='bow', metavar='M',
                    help='Method: [\'bow\', \'sift\', \'harris\'].')
args = parser.parse_args()

## Use the CATS dataset
list_img = glob.glob("../data/cats/*.jpg*")
assert len(list_img) > 0, "Put some images in the ./data/cats folder"
images_list = [load_input(im_name) for im_name in list_img]
fmap = load_input("./Figures/"+args.tname)

if (args.tmethod == "bow"):
	bow_comparison(fmap, images_list, name="cats", num_words=10, fmap_name="1", list_img=list_img)
if (arg.tmethod == "sift"):
	corresp_comparison(fmap, images_list, name="cats", fmap_name="1", list_img=list_img)
if (arg.tmethod == "harris"):
	repeatability_harris(fmap, images_list, name="cats", fmap_name="1", list_img=list_img)
