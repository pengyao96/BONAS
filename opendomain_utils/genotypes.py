from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

NASNet = Genotype(
    normal=[
        ('sep_conv_5x5', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 0),
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 0),
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
    ],
    normal_concat=[2, 3, 4, 5, 6],
    reduce=[
        ('sep_conv_5x5', 1),
        ('sep_conv_7x7', 0),
        ('max_pool_3x3', 1),
        ('sep_conv_7x7', 0),
        ('avg_pool_3x3', 1),
        ('sep_conv_5x5', 0),
        ('skip_connect', 3),
        ('avg_pool_3x3', 2),
        ('sep_conv_3x3', 2),
        ('max_pool_3x3', 1),
    ],
    reduce_concat=[4, 5, 6],
)

AmoebaNet = Genotype(
    normal=[
        ('avg_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 2),
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 3),
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 1),
    ],
    normal_concat=[4, 5, 6],
    reduce=[
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('max_pool_3x3', 0),
        ('sep_conv_7x7', 2),
        ('sep_conv_7x7', 0),
        ('avg_pool_3x3', 1),
        ('max_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('conv_7x1_1x7', 0),
        ('sep_conv_3x3', 5),
    ],
    reduce_concat=[3, 4, 6]
)


DARTS_V1 = Genotype(
    normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0),
            ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])
DARTS = DARTS_V2

BONAS = Genotype(
    normal=[('sep_conv_3x3', 0),
            ('sep_conv_3x3', 0),
            ('skip_connect', 0),
            ('max_pool_3x3', 0),
            ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 2),
            ('skip_connect', 1),
            ('sep_conv_3x3', 2)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('sep_conv_3x3', 0),
            ('sep_conv_3x3', 0),
            ('skip_connect', 0),
            ('max_pool_3x3', 0),
            ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 2),
            ('skip_connect', 1),
            ('sep_conv_3x3', 2)],
    reduce_concat=[2, 3, 4, 5])

#########################################
#log0-7 epoch 100
#log8-15 epoch 600
#0.9180000020751952 full train-100/600epoch // acc:92.42999770507812/95.65999753417968
exp_BONAS_0 = Genotype(
    normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 1), ('sep_conv_3x3', 3),
            ('skip_connect', 1), ('max_pool_3x3', 2), ('max_pool_3x3', 0)], normal_concat=[2, 3, 4, 5],
    reduce=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 1), ('sep_conv_3x3', 3),
            ('skip_connect', 1), ('max_pool_3x3', 2), ('max_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])

#0.9193000004882812 full train-100epoch // acc:93.73999787597656/95.80999760742188
exp_BONAS_1 = Genotype(
    normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('max_pool_3x3', 0), ('dil_conv_3x3', 3),
            ('dil_conv_3x3', 0), ('skip_connect', 0), ('max_pool_3x3', 2)], normal_concat=[2, 3, 4, 5],
    reduce=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('max_pool_3x3', 0), ('dil_conv_3x3', 3),
            ('dil_conv_3x3', 0), ('skip_connect', 0), ('max_pool_3x3', 2)], reduce_concat=[2, 3, 4, 5])

#0.9185999982910156full train-100epoch // acc:91.3699973876953/94.64999765625
exp_BONAS_2 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('dil_conv_3x3', 2),
            ('sep_conv_3x3', 1), ('max_pool_3x3', 1), ('max_pool_3x3', 3)], normal_concat=[2, 3, 4, 5],
    reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('dil_conv_3x3', 2),
            ('sep_conv_3x3', 1), ('max_pool_3x3', 1), ('max_pool_3x3', 3)], reduce_concat=[2, 3, 4, 5])

#0.9208999982910155 full train-100epoch // acc:93.09999763183593/95.78999729003907
exp_BONAS_3 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 1), ('max_pool_3x3', 2),('sep_conv_3x3', 3),
            ('max_pool_3x3', 2), ('dil_conv_3x3', 3), ('skip_connect', 0)],normal_concat=[2, 3, 4, 5],
    reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 1), ('max_pool_3x3', 2), ('sep_conv_3x3', 3),
            ('max_pool_3x3', 2), ('dil_conv_3x3', 3), ('skip_connect', 0)], reduce_concat=[2, 3, 4, 5])

#0.920099999267578 full train-100epoch // acc:93.20999736328125/95.69999780273437
exp_BONAS_4 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 1), ('sep_conv_3x3', 2), ('dil_conv_3x3', 0),
            ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 2)], normal_concat=[2, 3, 4, 5],
    reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 1), ('sep_conv_3x3', 2), ('dil_conv_3x3', 0),
            ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 2)], reduce_concat=[2, 3, 4, 5])

#0.9140999973144532full train-100epoch // acc:92.25999748535156/95.649997265625
exp_BONAS_5 = Genotype(
    normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 2), ('skip_connect', 2), ('max_pool_3x3', 2),
            ('skip_connect', 1), ('skip_connect', 4), ('max_pool_3x3', 0)], normal_concat=[2, 3, 4, 5],
    reduce=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 2), ('skip_connect', 2), ('max_pool_3x3', 2),
            ('skip_connect', 1), ('skip_connect', 4), ('max_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])

#0.9204000014648436 full train-100epoch // acc:94.74999748535156/96.35999787597656
exp_BONAS_6 = Genotype(
    normal=[('skip_connect', 0), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 1),('max_pool_3x3', 0),
            ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=[2, 3, 4, 5],
    reduce=[('skip_connect', 0), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 1), ('max_pool_3x3', 0),
            ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], reduce_concat=[2, 3, 4, 5])

#0.9193000014648436   full train-100epoch // acc:93.71999743652344/96.18999741210938
exp_BONAS_7 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('dil_conv_3x3', 4), ('max_pool_3x3', 1)], normal_concat=[2, 3, 4, 5],
    reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('dil_conv_3x3', 4), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])

exp_BONAS_8 = DARTS_V1   #log16- 96.15999755859374     log21-aux 96.5299975341796
exp_BONAS_9 = DARTS_V2    #log17- 96.02999721679687     log22-aux 96.55999775390625
exp_BONAS_10 = BONAS      #log18- 96.03999770507812     log23-aux 96.50999772949218
exp_BONAS_11 = NASNet   #log19- 96.209997              log24-aux 96.44999768066407
exp_BONAS_12 = AmoebaNet   #log20- 96.30999755859375     log25-aux 96.6199973876953

#0.9229000018310546
exp_BONAS_13 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 3),
            ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5],
    reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 3),
            ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 2)], reduce_concat=[2, 3, 4, 5])
#0.9261
exp_BONAS_14 = Genotype(
    normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('max_pool_3x3', 2), ('sep_conv_3x3', 2),
            ('max_pool_3x3', 3), ('dil_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5],
    reduce=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('max_pool_3x3', 2), ('sep_conv_3x3', 2),
            ('max_pool_3x3', 3), ('dil_conv_3x3', 0), ('skip_connect', 2)], reduce_concat=[2, 3, 4, 5])