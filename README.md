# GLSO: Grammar-guided Latent Space Optimization for Sample-efficient Robot Design Automation (CoRL 22)
Official implementation

## Building (Linux, Mac OS)

1. Set up the modified \[[RoboGrammar](https://github.com/JiahengHu/RoboGrammar.git)\] repo following the instructions.

2. Install baysian-optimization from github
* pip3 install git+https://github.com/fmfn/BayesianOptimization

3. Install required python packages for GLSO
* pip3 install -r requirements.txt

## Running Examples
### Step 1: collect training data for VAE
`cd robot_utils`; 
`python3 collect_data.py -i500000 --grammar_file {PATH_TO_ROBOGRAMMAR}/data/designs/grammar_apr30.dot`

### Step 2: train Graph VAE for design encoding
`python3 vae_train.py --save_dir sum_ls28_pred20 --data_dir new_train_data_loc_prune --gamma 20 
`

### Step 3: performing bayesian optimization in the latent space
`python3 run_bo.py --model sum_ls28_pred20 --task FlatTerrainTask --log_dir log --no_noise --rd_explore
`

### Step 4: visualize optimization results
Look into sample.py for different visualization options.