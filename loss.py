import torch


def calculateIOU(pstart, pend, start, end):
	start_min = min(pstart, start)
	end_max = max(pend, end)

	start_max = max(pstart, start)
	end_min = min(pend, end)

	if end_min < start_max:
		return 0

	IOU = (end_min - start_max) / (end_max - start_min)
	return IOU


def smoothl1loss(pstart, pend, start, end):
	loss = torch.abs(pstart-start) + torch.abs(start-end)
	if loss > 0.5:
		return loss
	else:
		return 0.5*loss*loss


def calculateLoss(predictions, labels, durations, attention_weights):
	batch_size = predictions.size(0)
	gt_weights = torch.zeros(attention_weights.size())

	loss = 0
	iou = 0
	for i in range(batch_size):
		start = labels[i][0]/durations[i]
		end = labels[i][1]/durations[i]
		pstart = predictions[i][0]
		pend = predictions[i][1]

		loss += smoothl1loss(pstart, pend, start, end)

		start *= durations[i]
		end *= durations[i]
		pstart *= durations[i]
		pend *= durations[i]
		iou += calculateLoss(pstart, pend, start, end)
	return loss, iou