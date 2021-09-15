import os

taskname = "supermodel_random_100"
local_root_dir = "./" # root working directory
local_data_dir = "./" # data root
results_dir = "trained_results"
trained_pickle_file = "trained_models.pkl"
trained_csv_file = "trained_models.csv"
logfile = 'BOGCN_open_domain.log'
io_config = dict(
    trained_pickle_file=os.path.join(local_root_dir, results_dir, taskname, trained_pickle_file),
    trained_csv_file=os.path.join(local_root_dir, results_dir, taskname, trained_csv_file),
)
search_config = dict(
    gcn_epochs=100,
    gcn_lr=0.001,
    loss_num=3,
    generate_num=500,    #随机生成一个大池子，然后从中细选
    iterations=12,
    bo_sample_num=100,  #supernet 一次性覆盖的样本数
    sample_method="random",
    if_init_samples=True,
    init_num=100,   #要求随机初始化的样本数
)

training_config = dict(
    train_supernet_epochs=100,
    data_path=os.path.join(local_data_dir, 'data'),
    super_batch_size=64,
    sub_batch_size=128,
    learning_rate=0.025,
    momentum=0.9,
    weight_decay=3e-4,
    report_freq=50,
    epochs=100,
    init_channels=36,
    layers=20,
    drop_path_prob=0.2,
    seed=0,
    grad_clip=5,
    parallel=False,
    mode='random'
)

distributed = False

#OPS to allow in the search space
OPS = ['input', 'max_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'dil_conv_3x3', 'output']
