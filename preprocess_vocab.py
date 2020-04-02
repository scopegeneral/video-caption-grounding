import re
import json
import argparse
import numpy as np


def build_vocab(vids, params):
		
	count_thr = params['word_count_threshold']
	# count up the number of words
	counts = {}
	for vid in vids:
		for cap in vids[vid]['sentences']:
			ws = re.sub(r'[.!,;?]', ' ', cap).split()
			for w in ws:
				counts[w] = counts.get(w, 0) + 1
	# cw = sorted([(count, w) for w, count in counts.items()], reverse=True)
	total_words = sum(counts.values())
	bad_words = [w for w, n in counts.items() if n <= count_thr]
	vocab = [w for w, n in counts.items() if n > count_thr]
	bad_count = sum(counts[w] for w in bad_words)
	print('number of bad words: %d/%d = %.2f%%' %
		  (len(bad_words), len(counts), len(bad_words) * 100.0 / len(counts)))
	print('number of words in vocab would be %d' % (len(vocab), ))
	print('number of UNKs: %d/%d = %.2f%%' %
		  (bad_count, total_words, bad_count * 100.0 / total_words))
	# lets now produce the final annotations
	if bad_count > 0:
		# additional special UNK token we will use below to map infrequent words to
		print('inserting the special UNK token')
		vocab.append('<UNK>')
	
	
	newvids = {}
	for vid in vids:
		newvids[vid] = {}
		newvids[vid]['sentences'] = []
		newvids[vid]['timestamps'] = vids[vid]['timestamps']
		newvids[vid]['duration'] = vids[vid]['duration']
		newvids[vid]['subset'] = vids[vid]['subset']
		for cap in vids[vid]['sentences']:
			ws = re.sub(r'[.!,;?]', ' ', cap).split()
			caption = ['<sos>'] + [w if counts.get(w, 0) > count_thr else '<UNK>' for w in ws] + ['<eos>']
			newvids[vid]['sentences'].append(caption)
			
	json.dump(newvids, open(params['processed_json'],'w'))
	return vocab


def main(params):

	words = []
	idx = 0
	word2idx = {}
	vectors = bcolz.carray(np.zeros(1), rootdir=f'{glove_path}/6B.50.dat', mode='w')

	with open(f'{glove_path}/glove.6B.50d.txt', 'rb') as f:
		for l in f:
			line = l.decode().split()
			word = line[0]
			words.append(word)
			word2idx[word] = idx
			idx += 1
			vect = np.array(line[1:]).astype(np.float)
			vectors.append(vect)
		
	vectors = bcolz.carray(vectors[1:].reshape((400000, 50)), rootdir=f'{glove_path}/6B.50.dat', mode='w')
	vectors.flush()
	pickle.dump(words, open(f'{glove_path}/6B.50_words.pkl', 'wb'))
	pickle.dump(word2idx, open(f'{glove_path}/6B.50_idx.pkl', 'wb'))

	vectors = bcolz.open(f'{glove_path}/6B.50.dat')[:]
	words = pickle.load(open(f'{glove_path}/6B.50_words.pkl', 'rb'))
	word2idx = pickle.load(open(f'{glove_path}/6B.50_idx.pkl', 'rb'))

	glove = {w: vectors[word2idx[w]] for w in words}



	videos = json.load(open(params['input_json'], 'r'))
	# create the vocab
	vocab = build_vocab(videos, params)

	matrix_len = len(vocab)
	weights_matrix = np.zeros((matrix_len+3, emb_dim))
	words_found = 0

	weights_matrix[0] = np.random.normal(scale=0.6, size=(emb_dim, ))
	weights_matrix[1] = np.random.normal(scale=0.6, size=(emb_dim, ))
	weights_matrix[2] = np.random.normal(scale=0.6, size=(emb_dim, ))

	for i, word in enumerate(target_vocab):
		try: 
			weights_matrix[i+3] = glove[word]
			words_found += 1
		except KeyError:
			weights_matrix[i+3] = np.random.normal(scale=0.6, size=(emb_dim, ))

	np.save('save_filename_for_glove_weights', weights_matrix)

	itow = {i + 3: w for i, w in enumerate(vocab)}
	wtoi = {w: i + 3 for i, w in enumerate(vocab)}  
	wtoi['<pad>'] = 0
	itow[0] = '<pad>'
	wtoi['<eos>'] = 1
	itow[1] = '<eos>'
	wtoi['<sos>'] = 2
	itow[2] = '<sos>'

	out = {}
	out['ix_to_word'] = itow
	out['word_to_ix'] = wtoi
	out['videos'] = {'train': [], 'test': []}
	#videos = json.load(open(params['input_json'], 'r'))['database']

	final_idx = {}
	labels = {}
	idx = 0

	videos = json.load(open(params['processed_json'], 'r'))

	for i in videos:
		if videos[i]['subset'] == 'testing':
			continue
		for timestamps_idx, cap in enumerate(videos[i]['sentences']):
			final_idx[idx] = {'idx':idx, 'video':i, 'caption':cap}
			labels[idx] = {'idx':idx, 'duration':videos[i]['duration'], 'label':videos[i]['timestamps'][timestamps_idx]}

			if videos[i]['subset'] == 'training':
				out['videos']['train'].append(idx)
				
			if videos[i]['subset'] == 'validation':
				out['videos']['test'].append(idx)

			idx += 1


	json.dump(out, open(params['info_json'], 'w'))
	json.dump(final_idx, open(params['caption_json'],'w'))
	json.dump(labels, open(params['timestamps_json'],'w'))
	

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	# input json
	parser.add_argument('--input_json', type=str, default='captions/activity_net.json', \
						help='activity_net videoinfo json')
	
	parser.add_argument('--info_json', default='data/info.json', help='info about iw2word and word2ix')
	
	parser.add_argument('--caption_json', default='data/caption.json', help='final captions json file')
	
	parser.add_argument('--processed_json', default='data/processed.json', help='cleaner captions json file')

	parser.add_argument('--timestamps_json', default='data/timestamps.json', help='starttime and endtimes')

	parser.add_argument('--word_count_threshold', default=1, type=int, \
						help='only words that occur more than this number of times will be put in vocab')

	args = parser.parse_args()
	params = vars(args)  # convert to ordinary dict
	main(params)