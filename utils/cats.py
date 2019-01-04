#coding:utf-8

from imagenet1000 import imagenet1000

def get_label(name):
	labels = imagenet1000.values()
	return imagenet1000.keys()[labels.index(name)]

dict_labels_cats = {'./data/cats/cat8.jpg': get_label('tiger cat'), 
	'./data/cats/cat7.jpg': get_label('tiger cat'), 
	'./data/cats/cat6.jpg': get_label('tiger cat'), 
	'./data/cats/cat3.jpg': get_label('tabby, tabby cat'), 
	'./data/cats/cat1.jpg': get_label('Persian cat'), 
	'./data/cats/cat4.jpg': get_label('tabby, tabby cat'), 
	'./data/cats/cat5.jpg': get_label('Siamese cat, Siamese'), 
	'./data/cats/cat9.jpg': get_label('Egyptian cat'), 
	'./data/cats/cat11.jpg': get_label('tiger cat'), 
	'./data/cats/cat2.jpg': get_label('tiger cat'), 
	'./data/cats/cat10.jpg': get_label('tabby, tabby cat')
}
