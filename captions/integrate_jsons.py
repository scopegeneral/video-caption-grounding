import json

final_json = {}

train = json.load(open('train.json','r'))
val_1 = json.load(open('val_1.json','r'))
val_2 = json.load(open('val_2.json','r'))

for i in train:
	final_json[i] = train[i]
	final_json[i]['subset'] = 'training'

for i in val_1:
	final_json[i] = val_1[i]
	final_json[i]['subset'] = 'validation'

for i in val_2:
	final_json[i] = val_2[i]
	final_json[i]['subset'] = 'validation'

json.dump(final_json, open('activity_net.json', 'w'))