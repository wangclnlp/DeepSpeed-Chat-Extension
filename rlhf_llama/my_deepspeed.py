import os

from deepspeed.launcher.runner import main

os.environ["PATH"] = os.environ["PATH"] + ":/root/miniconda3/bin/"
# os.environ['HF_DATASETS_OFFLINE'] = '1'


if __name__ == '__main__':
    main()
