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


exp_BONAS_8 = DARTS_V1  # log16- 96.15999755859374     log21-aux 96.5299975341796
exp_BONAS_9 = DARTS_V2  # log17- 96.02999721679687     log22-aux 96.55999775390625
exp_BONAS_10 = BONAS  # log18- 96.03999770507812     log23-aux 96.50999772949218
exp_BONAS_11 = NASNet  # log19- 96.209997              log24-aux 96.44999768066407
exp_BONAS_12 = AmoebaNet  # log20- 96.30999755859375     log25-aux 96.6199973876953

# 0.9229000018310546
xx_BONAS_0 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 3),
            ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5],
    reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 3),
            ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 2)], reduce_concat=[2, 3, 4, 5])
# 0.9261
xx_BONAS_1 = Genotype(
    normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('max_pool_3x3', 2), ('sep_conv_3x3', 2),
            ('max_pool_3x3', 3), ('dil_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5],
    reduce=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('max_pool_3x3', 2), ('sep_conv_3x3', 2),
            ('max_pool_3x3', 3), ('dil_conv_3x3', 0), ('skip_connect', 2)], reduce_concat=[2, 3, 4, 5])



raw_BONAS_0 = Genotype(
    normal=[('dil_conv_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 1),
            ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 3)], normal_concat=[2, 3, 4, 5],
    reduce=[('dil_conv_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 1),
            ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 3)], reduce_concat=[2, 3, 4, 5])
# raw_BONAS_1 = Genotype(
#     normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
#             ('dil_conv_3x3', 1), ('sep_conv_3x3', 1), ('max_pool_3x3', 1)], normal_concat=[2, 3, 4, 5],
#     reduce=[('sep_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
#             ('dil_conv_3x3', 1), ('sep_conv_3x3', 1), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])
# raw_BONAS_2 = Genotype(
#     normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
#             ('sep_conv_3x3', 1), ('max_pool_3x3', 1), ('skip_connect', 1)], normal_concat=[2, 3, 4, 5],
#     reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
#             ('sep_conv_3x3', 1), ('max_pool_3x3', 1), ('skip_connect', 1)], reduce_concat=[2, 3, 4, 5])
# raw_BONAS_3 = Genotype(
#     normal=[('skip_connect', 0), ('dil_conv_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 1),
#             ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_3x3', 3)], normal_concat=[2, 3, 4, 5],
#     reduce=[('skip_connect', 0), ('dil_conv_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 1),
#             ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_3x3', 3)], reduce_concat=[2, 3, 4, 5])
