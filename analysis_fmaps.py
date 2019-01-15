# coding:utf-8
import sys
sys.path += ['layers', 'models', 'utils']
import argparse
import subprocess as sb
from comp_fmaps import bow_comparison, corresp_comparison, repeatability_harris, load_input

## python3.6 analysis_fmaps.py --tname conv/convfeature_map_layer_conv1.png --tdata CATS --tmethod bow

## python3.6 analysis_fmaps.py --tmodel conv --texperiment outputs --tlayer {conv1|conv2|...|conv6} --tdata CATS --tmethod bow
## python3.6 analysis_fmaps.py --tmodel conv --texperiment reconst --tdata siamese --tmethod bow --ab {a|b} --nb {1|...|10}

parser = argparse.ArgumentParser(description='Analysis of deconvoluted images')
parser.add_argument('--tname', type=str, default='', metavar='N',
                    help='Name of image to analyze.')
parser.add_argument('--tmodel', type=str, default='', metavar='M',
                    help='Name of model to analyze (if --tname is not specified). In ["vgg", "conv"]')
parser.add_argument('--texperiment', type=str, default='', metavar='E',
                    help='Name of experiment to run (if --tname is not specified). In ["outputs", "reconst"]')
parser.add_argument('--nb', type=str, default='', metavar='B',
                    help='Number of experiment for texperiment==reconst (if --tname is not specified). Similar to the argument "nb" in process_model.py')
parser.add_argument('--ab', type=str, default='', metavar='A',
                    help='Before/after training for texperiment==reconst (if --tname is not specified). In ["a", "b"]')
parser.add_argument('--tlayer', type=str, default='', metavar='L',
                    help='Layer to analyze for texperiment==outputs (if --tname is not specified).')
parser.add_argument('--tdata', type=str, default='CATS', metavar='D',
                    help='Dataset to use ["CATS", "siamese"].')
parser.add_argument('--tmethod', type=str, default='bow', metavar='H',
                    help='Method: [\'bow\', \'sift\', \'harris\'].')
args = parser.parse_args()

if (not args.tname and args.tmodel and args.texperiment):
	name = "exp/exp_"+args.tmodel+"/"+args.texperiment+"/"
	if (args.texperiment=="outputs"):
		name += "convfeature_map_layer_"+args.tlayer+".png"
	if (args.texperiment=="reconst"):
		class_ = str(284) ## Same as in process_model.py in the final pipeline
		name += "conv_feature_map_layer"+args.ab+"_training_deconv_class="+class_+"_"+args.nb+".png"
else:
	assert args.tname != None, "Please specify the name of the input image"
	name = args.tname

## Load images
if (args.tdata == "CATS"):
	data = "cats"
	title = "cat"
if (args.tdata == "siamese"):
	data = "siamese"
	title = "siamese"
list_img = ["./data/"+data+"/"+title+str(i)+".jpg" for i in range(1, 12)]
assert len(list_img) > 0, "Put some images in the ./data/"+data+" folder"
images_list = [load_input(im_name) for im_name in list_img]
fmap = load_input("./Figures/"+name)

ntry = 1

if (args.tmethod == "bow"):
	bow_comparison(fmap, images_list, name=data, num_words=10, fmap_name=str(ntry), list_img=list_img)
	sb.call("mv ./bow_input_closest.png ./slides+report/figures_/bow_analysis/", shell=True)
	sb.call("mv ./image_input_closest.png ./slides+report/figures_/bow_analysis/", shell=True)
	sb.call("mv ./data/bow_sift_comp/bow/bow_"+data+"_1_scores.dat ./slides+report/figures_/bow_analysis/", shell=True)
if (args.tmethod == "sift"):
	corresp_comparison(fmap, images_list, name=data, fmap_name=str(ntry), list_img=list_img)
	sb.call("mv ./data/bow_sift_comp/corresp/corresp_"+data+"_1_contributions.dat ./slides+report/figures_/sift_analysis/", shell=True)
if (args.tmethod == "harris"):
	repeatability_harris(fmap, images_list, name=data, fmap_name=str(ntry), list_img=list_img)
	sb.call("mv ./data/bow_sift_comp/harris/harris_"+data+"_1_contributions.dat ./slides+report/figures_/harris_analysis/", shell=True)
