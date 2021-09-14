'''
To run search algorithm on 8 gpus separately.
This reduces communication time between gpus comparing to DataParallel and Distributed Dataparallel
To search on a dataset such as imagenet, which requires more memory, DDP is expected to be implemented
'''
import os
import subprocess

import torch

from opendomain_utils.ioutils import create_dirs
from settings import local_root_dir

create_dirs()

if torch.cuda.is_available():
    python_path = 'python'  # path of python command
    gpu_num = 8
    tasks = []
    for gpu in range(gpu_num):
        cmd = [python_path, os.path.join(local_root_dir, 'BOGCN_opendomain.py'), f"--gpu={gpu}"]
        train = subprocess.Popen(cmd)
        tasks.append(train)
    for task in tasks:
        task.wait()
    print(f"Num of Tasks finished:{len(tasks)}")
else:
    cmd = 'python BOGCN_opendomain.py --gpu 0'
    os.system(cmd)
