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

# python train_cifar10.py --gpu=1 --arch=raw_BONAS_0 --save=raw_1 --seed=0 --auxiliary
# python train_cifar10.py --gpu=2 --arch=raw_BONAS_0 --save=raw_2 --seed=0

# python train_cifar10.py --gpu=3 --arch=raw_BONAS_0 --save=raw_3 --seed=0 --epochs 1200 --auxiliary  --cutout
# python train_cifar10.py --gpu=4 --arch=raw_BONAS_0 --save=raw_4 --seed=0 --batch_size 128 --auxiliary  --cutout  #97.03

# python train_cifar10.py --gpu=5 --arch=raw_BONAS_0 --save=raw_5 --seed=11 --auxiliary  --cutout  #97.2399976074218
# python train_cifar10.py --gpu=6 --arch=raw_BONAS_0 --save=raw_6 --seed=123 --auxiliary  --cutout
# python train_cifar10.py --gpu=7 --arch=raw_BONAS_0 --save=raw_7 --seed=1233 --auxiliary  --cutout

# python train_cifar10.py --gpu=0 --arch=raw_BONAS_0 --save=raw_0 --seed=0 --auxiliary  --cutout  #97.15999760742187
# python train_cifar10.py --gpu=0 --arch=raw_BONAS_0 --save=raw_8 --seed=0 --batch_size 128 --epochs 1200 --auxiliary  --cutout  #97.36

# python train_cifar10.py --gpu=2 --arch=xx_BONAS_0 --save=xx_0 --seed=11 --auxiliary  --cutout   #97.20999719238282
# python train_cifar10.py --gpu=3 --arch=xx_BONAS_1 --save=xx_1 --seed=11 --auxiliary  --cutout    #95.6099971435547
# python train_cifar10.py --gpu=4 --arch=xx_BONAS_0 --save=xx_3 --seed=11 --batch_size 128 --epochs 1200 --auxiliary  --cutout  #97.21   2.64M

# python train_cifar10.py --gpu=5 --arch=raw_BONAS_1 --save=xx_4 --seed=11 --auxiliary  --cutout  #97.28999716796875
# python train_cifar10.py --gpu=6 --arch=raw_BONAS_2 --save=xx_5 --seed=11 --auxiliary  --cutout   #97.19999743652343
# python train_cifar10.py --gpu=7 --arch=raw_BONAS_3 --save=xx_6 --seed=11 --auxiliary  --cutout     #97.2899974609375


# python train_cifar10.py --gpu=0 --arch=raw_BONAS_0 --save=raw_8 --seed=0 --batch_size 128 --epochs 1200 --auxiliary  --cutout  #97.36 /97.31      3.45M
# python train_cifar10.py --gpu=5 --arch=raw_BONAS_1 --save=xx_7 --seed=11 --batch_size 128 --epochs 1200 --auxiliary  --cutout   #97.52 /97.57     3.29M
# python train_cifar10.py --gpu=6 --arch=raw_BONAS_2 --save=xx_8 --seed=11 --batch_size 128 --epochs 1200 --auxiliary  --cutout    #97.31 /97.54    3.47M
# python train_cifar10.py --gpu=7 --arch=raw_BONAS_3 --save=xx_9 --seed=11 --batch_size 128 --epochs 1200 --auxiliary  --cutout       #97.63/97.46  3.06M