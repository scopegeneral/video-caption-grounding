#look at mask in model
#implement loss 
import torch
import json
from torch.utils import data
from model import ABLR
from dataloader import VideoDataset, PadCollate
import options
from loss import calculateLoss

opts = options.parse_opts()
opts = vars(opts)







use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
cudnn.benchmark = True

dataload_params = {'batch_size': 100,
				   'shuffle': True,
				   'num_workers': 10}

max_epochs = 200

partition = json.load(open('data/info.json','r'))

training_set = Dataset(partition['videos']['train'])
training_generator = data.DataLoader(training_set, **params, collate_fn=PadCollate(dim=0))

testing_set = Dataset(partition['videos']['test'])
testing_generator = data.DataLoader(testing_set, **params, collate_fn=PadCollate(dim=0))

Model = ABLR(opts).to(device)
optimizer = optim.Adam( model.parameters(),
						lr=opts["learning_rate"])
						#weight_decay=opt["weight_decay"])

for epoch in range(max_epochs):
	iteration = 0
	iou = 0
	# Training
	for local_batch in training_generator:
		# Transfer to GPU
		video_feats = local_batch['video_feat'].to(device)
		caption_feats = local_batch['caption_feat'].to(device)
		masks = local_batch['masks'].to(device)
		labels = local_batch['label'].to(device)
		durations = local_batch['duration'].to(device)

		attention_weights, predictions = Model(opts, video_feats, caption_feats, masks)
		loss, iou1 = calculateLoss(predictions, labels, durations, attention_weights)
		iou += iou1
		loss.backward()
		clip_grad_value_(model.parameters(), opts['grad_clip'])
		optimizer.step()
		train_loss = loss.item()
		iteration += 1
		print("iter %d (epoch %d), train_loss = %.6f, iou = %.6f" % (iteration, epoch, train_loss, iou1/opts.batch_size))
	
	# Testing
	if epoch % opts["save_checkpoint_every"] == 0:
		with torch.set_grad_enabled(False):
			iteration = 0
			iou = 0
			for local_batch in testing_generator:
				# Transfer to GPU
				video_feats = local_batch['video_feat'].to(device)
				caption_feats = local_batch['caption_feat'].to(device)
				masks = local_batch['masks'].to(device)
				labels = local_batch['label'].to(device)
				durations = local_batch['duration'].to(device)

				attention_weights, predictions = Model(opts, video_feats, caption_feats, masks)
				loss, iou1 = calculateLoss(predictions, labels, durations, attention_weights)
				iou += iou1
				test_loss = loss.item()
				iteration += 1
				print("iter %d (epoch %d), test_loss = %.6f iou=%0.6f" % (iteration, epoch, test_loss, iou1/opts.batch_size))




		model_path = os.path.join(opts["checkpoint_path"], 'model_%d.pth' % (epoch))
		model_info_path = os.path.join(opts["checkpoint_path"], 'model_score.txt')
		torch.save(model.state_dict(), model_path)
		print("model saved to %s" % (model_path))
		with open(model_info_path, 'a') as f:
			f.write("model_%d, loss: %.6f\n" % (epoch, train_loss))

