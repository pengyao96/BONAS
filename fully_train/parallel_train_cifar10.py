'''
To run search algorithm on 8 gpus separately.
This reduces communication time between gpus comparing to DataParallel and Distributed Dataparallel
To search on a dataset such as imagenet, which requires more memory, DDP is expected to be implemented
'''
import os
import subprocess

python_path = 'python'  # path of python command
gpu_num = 8
tasks = []

for gpu in range(gpu_num):
    cmd = [python_path, os.path.join('./', 'train_cifar10.py'), f"--gpu={gpu}", f"--arch=exp_BONAS_{gpu}",
           f"--save={gpu}"]
    train = subprocess.Popen(cmd)
    tasks.append(train)
for task in tasks:
    task.wait()
print(f"Num of Tasks finished:{len(tasks)}")
