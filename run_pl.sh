#!/bin/bash


#python pipelineNew.py --hide_window=true --weights_file='_5_5_TransferLSTM_TS' --test_mode=false --bayesian=false --sourceDir_path='./web_server'

#python pipelineNew.py hide_window=true --weights_file='_5_5_TransferLSTM_TS' test_mode=false bayesian=false --sourceDir_path='./web_server'

python pipelineNew.py --weights_file="_5_5_TransferLSTM_TS"  --sourceDir_path="web_server/"
