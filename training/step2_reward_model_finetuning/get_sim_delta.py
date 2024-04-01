import json

json_file = 'models/alpaca/reward/len_rat/beta1_10_beta3_-5/epoch-1/train_analyze/similarity_length_ratio.json'

output_file = 'len_rat-beta1_10_beta3_-5-delta-sim_len_rat.dat'


with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)
    data1 = [d for d in data if d['delta_score']>0]
    data0 = [d for d in data if d['delta_score']<=0]

with open(output_file, 'w', encoding='utf-8') as f:
    # f.write('{}\t{}'.format(data1[0]['similarity'], data1[0]['delta_score']))
    f.write('sim\tdelta\tlabel')
    for d in data1:
        f.write('\n{}\t{}\t{}'.format(d['similarity'], d['delta_score'], 'r'))
    # for d in data0:
    #     f.write('\n{}\t{}\t{}'.format(d['similarity'], d['delta_score'], 'w'))

