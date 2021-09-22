'''
To run search algorithm on 8 gpus separately.
This reduces communication time between gpus comparing to DataParallel and Distributed Dataparallel
To search on a dataset such as imagenet, which requires more memory, DDP is expected to be implemented
'''
import os
import subprocess
import random
python_path = 'python'  # path of python command
gpu_num = 4
tasks = []

for gpu in range(gpu_num):
    # seed = random.randint(1,10000)
    cmd = [python_path, os.path.join('./', 'train_cifar10.py'), f"--gpu={gpu}", f"--arch=raw_BONAS_{gpu}",
           f"--save=raw_{gpu}", "--auxiliary", f"--cutout"]
    train = subprocess.Popen(cmd)
    tasks.append(train)
for task in tasks:
    task.wait()
print(f"Num of Tasks finished:{len(tasks)}")




#python train_cifar10.py --gpu=0 --arch=exp_BONAS_13 --save=34 --auxiliary --seed=123