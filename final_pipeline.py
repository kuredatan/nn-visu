#coding: utf-8
import sys
sys.path += ['layers', 'models', 'utils']
import argparse
import subprocess as sb

parser = argparse.ArgumentParser(description='Analysis of deconvoluted images')
parser.add_argument('--tmodel', type=str, default='conv', metavar='M',
                    help='CNN to visualize/analyze ["conv", "vonc", "conv2", "vgg"].')
parser.add_argument('--ntry', type=int, default=10, metavar='N',
                    help='Number of experiments for analyzing of training process (see report).')
parser.add_argument('--start', type=int, default=0, metavar='N',
                    help='Start index of experiments for analyzing of training process (see report). (optional)')
parser.add_argument('--tmethod', type=str, default='bow', metavar='H',
                    help='Method: [\'bow\', \'sift\', \'harris\'].')
parser.add_argument('--py', type=str, default='2.7', metavar='P',
                    help='Python version for which Keras and Tensorflow have been installed: [\'2.7\', \'3.6\', ...].')
args = parser.parse_args()

if (args.start):
	rang = range(args.start, args.ntry+args.start)
else:
	rang = range(1, args.ntry+1)

for i in rang:
	calls = []
	ncalls = ["GENERATING INPUTS #"+str(i), "ANALYSIS BEFORE TRAINING", "ANALYSIS AFTER TRAINING"]
	calls.append("python"+args.py+" process_model.py --tmodel "+args.tmodel+" --trained 1 --trun final --batch 32 --tdata siamese --lr 0.001 --optimizer Adam --loss categorical_crossentropy --epoch 10 --all 1 --nb "+str(i)+" --step 1")
	calls.append("python3.6 analysis_fmaps.py --tmodel "+args.tmodel+" --texperiment reconst --tdata siamese --tmethod "+args.tmethod+" --ab b --nb "+str(i))
	calls.append("python3.6 analysis_fmaps.py --tmodel "+args.tmodel+" --texperiment reconst --tdata siamese --tmethod "+args.tmethod+" --ab a --nb "+str(i))
	for i in range(len(calls)):
		m = len(ncalls[i])+6
		print("#"*m)
		print("## " + ncalls[i] + " ##")
		print("#"*m)
		print("CALL: \"" + calls[i] + "\"")
		sb.call(calls[i], shell=True)

msg = "PLOTS/RESULTS"
m = len(msg)+6
print("#"*m)
print("## "+msg+" ##")
print("#"*m)
call = "python ./utils/plot_contrib.py --tmethod "+args.tmethod+" --tdata siamese --tmodel "+args.tmodel
print("CALL: \"" + call + "\"")
sb.call(call, shell=True)
import gc
gc.collect()
