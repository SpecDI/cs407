# Workflow
This doc will show you how to setup and use tensorflow-keras for running the action recognition networks.

## 1. Setup
### 1.1 Directories
You should have a directory structure like this:
```
CS407
|
| - Data
|     |- single_label (Okutama)
|
| - Frame_Generators
|     |- VideoFrameGenerator_v2_cutoff
|
| - Action_Recognition_Networks
|     |- one_stream_3D_cnn.py
```
### 1.2 Imports
**NOTE: You need to use python3.6 because of tensorflow-gpu**

I am assuming that your ~/.local/lib/python3.6 folder is empty.

First run:
```
export LD_LIBRARY_PATH=/local/java/cuda_9.0/lib64/:/local/java/cudnn-7.1_for_cu a_9.0/lib64/
```

Then you must ``pip3`` install the following modules:
- ```tensorflow-gpu (1.14.0)```
- ```numpy(1.14.5)```
- ```keras(2.3.1)```
- ```matplotlib(3.1.1)```
- ```pillow (6.2.1)```
- ```scipy (1.3.2)```

## 2. Running the code
Run the ```one_stream_3D_cnn``` file in the following way:
```
python3 one_stream_3D_cnn
```

Check how much resources it is using via the command ```htop```

After the code has finished, check the results using tensorboard by doing the following:

```
python3 main.py --logdir=$HOME/.../CS407/Action_Recognition_Networks/logs
```

Then open the local-host.

## 3. Remote access

If you want to do all this remotely, ssh into one of the stone or cobra machines from the following list:
https://warwick.ac.uk/fac/sci/dcs/intranet/user_guide/compute_nodes/
