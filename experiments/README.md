# Experiments

Because of copyright restrictions, we are unable to provide the dataset. Consequently, the subsequent steps can only be performed by CSCS users who have the necessary permissions. However, the model weights can be found in the `outputs` directory, and these can be used to generate samples with the notebooks referenced below.

1. To build the dataset, use the script `build_dataset.sh`. The resulting train/test split will be stored in the `datasets` folder.

2. To train the classifier, run the script `train_eval_classifier.sh`. Relevant parameters can be configured by modifying the correspondig configuration file `configs/classifier.py`

3. To train the diffusion model, run the script `train_gm0_diffusion.sh`. Two configuration files are available: `ddim_1d.py` and `ddim_2d.py`. 

4. Evaluation: 
    - The notebook `notebooks/GM0-evaluation_TEMPLATE.ipynb` provides a full pipeline for the evaluation of the models. In particular `notebooks/GM0-evaluation_ME.ipynb` and `notebooks/GM0-evaluation_MS.ipynb` are the precompiled notebooks used for the evaluation of the two proposed models. Please refer to the instructions inside the notebook, and to the documentations of the methods used. 
    - The script `evaluate.sh` can be used to generate all the files needed for a possible evaluation. However, a notebook that takes as input the results of that script is **not implemented yet**. 