# Workflow

This doc will show you how to create and train keras models easily using the `TrainingSuite` class in `training.py`.

## 1. Directory Structure

The structure of this directory is as follows:

- `logs` directory contains the tensorboard log files for training a specific model.
- `weights` directory contains the best weights of each model
- `training.py` contains code for loading in action tubes, training and evaluating your model. 
- `<Model>.py` these are files that contain the definition of your Keras model (e.g. `_1_0_LSTM_OS.py`)

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

## 2. Defining your Model and using TrainingSuite 
You should create a new model file in the `architectures` directory. Try to stick to the naming convention: `_<version>_<Model_Type>_<Stream_Type>.py`.

See `_1_0_LSTM_OS.py` for an example of how to define your model.

In this file you should only need to set constants such as batch_size, epochs, kernel_size (some of which should be found by grid searching) and define the method for the model itself.

After defining the model, you should create a `TrainingSuite` object for your model and use its `evaluation` method to train and evalue your model.

For example:
`training_suite.evaluation(model, WEIGHT_FILE_NAME)` 

**IMPORTANT: The WEIGHT_FILE_NAME should be unique to the model so that you don't overwrite other weight files.**


## 3. Running the code

**Do not run the training.py code**

Run your `<Model>.py` file for training and evaluating your model.

```
python3 _1_0_LSTM_OS.py
```

Check how much resources it is using via the command ```htop```

After the code has finished, check the results using tensorboard by doing the following:

```
cd .local/lib/python3.6/site-packages/tensorboard

python3 main.py --logdir=<path to architectures>/architectures/logs
```

Then open the local-host.

## 3. Remote access

If you want to do all this remotely, ssh into one of the stone or cobra machines from the following list:
https://warwick.ac.uk/fac/sci/dcs/intranet/user_guide/compute_nodes/
