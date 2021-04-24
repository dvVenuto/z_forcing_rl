Install TF-AGENTS
Install https://github.com/tensorflow/agents/ version r0.7.1 under a virtual enviorment with package versions:

tensorflow-probability 0.12.1

tensorflow 2.4.0

dm-env 1.4

gym 0.18.0

mujoco_py 2.0.2.13

tensorflow-estimator 2.4.0

dm-tree 0.1.5

dm-control 0.0.355168290

Modify TF agents installation with these updated files
AGENTS_PATH refers to your instllation of tf-agents 0.7.1 .

Move init.py to /AGENTS_PATH/networks

Move lstm_encoding_network.py to /AGENTS_PATH/networks

Move LSTM_cell_test.py to /AGENTS_PATH/networks

Move GroupLinearLayer.py to /AGENTS_PATH/networks

Move GroupLSTMCell.py to /AGENTS_PATH/networks

Move RIMCell.py to /AGENTS_PATH/networks

Move RIMCellPlay.py to /AGENTS_PATH/networks

Move actor_rnn_network.py to /AGENTS_PATH/agents/ddpg
