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

# python train_cifar10.py --gpu=0 --arch=raw_BONAS_0 --save=raw_0 --seed=0 --auxiliary  --cutout
# python train_cifar10.py --gpu=1 --arch=raw_BONAS_0 --save=raw_1 --seed=0 --auxiliary
# python train_cifar10.py --gpu=2 --arch=raw_BONAS_0 --save=raw_2 --seed=0

# python train_cifar10.py --gpu=3 --arch=raw_BONAS_0 --save=raw_3 --seed=0 --epochs 1200 --auxiliary  --cutout
# python train_cifar10.py --gpu=4 --arch=raw_BONAS_0 --save=raw_4 --seed=0 --batch_size 128 --auxiliary  --cutout

# python train_cifar10.py --gpu=5 --arch=raw_BONAS_0 --save=raw_5 --seed=11 --auxiliary  --cutout
# python train_cifar10.py --gpu=6 --arch=raw_BONAS_0 --save=raw_6 --seed=123 --auxiliary  --cutout
# python train_cifar10.py --gpu=7 --arch=raw_BONAS_0 --save=raw_7 --seed=1233 --auxiliary  --cutout


# python train_cifar10.py --gpu=0 --arch=raw_BONAS_0 --save=raw_8 --seed=0 --batch_size 128 --epochs 1200 --auxiliary  --cutout

# python train_cifar10.py --gpu=2 --arch=xx_BONAS_0 --save=raw_0 --seed=123 --auxiliary  --cutout
# python train_cifar10.py --gpu=3 --arch=xx_BONAS_0 --save=raw_0 --seed=123 --auxiliary  --cutout
