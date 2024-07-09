# Spotting Toxic Clouds Thesis
 In this repo you can find the source code to the thesis: **Harnessing Self-Supervised Learning for Environmental Monitoring: Detecting Smoke in Industrial Cloud Formations** 

 ## Data Infrastructure
 Data was sourced from: https://github.com/CMU-CREATE-Lab/deep-smoke-machine by following their instructions.
 - Unlabeled data can be found in 'full_dataset'
 - Labeled data can be found in the 'train_set', 'validation_set', 'test_set'
 - Split metadata files can be found in 'split'

  ## Code Infrastructure
  - train.py: Contains the code for the implementation of SimCLR on unlabeled data
  - baseline.py: Contains the code for the baseline supervised model
  - lineareval.py: Contains the code for the fine-tuned self-supervised model
  - eda_process.ipynb: Contains the code for plotting label distribution and creating example (transformed) frame figures
  - metadata_02242020.json: Metadata file for the labeled data, sourced from CMU-CREATE-Lab GitHub
  - split_videos.py: Contains the code for splitting the videos into separate folders

  ## Set-up
  1. **Computing resources and Packages**: For this thesis, the Snellius supercomputer was used. Therefore, an environment had to be installed that included Python and Conda etc. If you already have those installations, feel free to only install the necessary modules. You can install dependencies from the 'dl2023_gpu.yml' environment file, which was sourced from: https://github.com/uvadlc/uvadlc_practicals_2023/tree/main.
  2. **Install data from source**: Data was installed from the CMU-CREATE-Lab. A folder called 'video' will be created.
  3. **Split data**: Metadata was split by using the 'split_metadata.py' file from the CMU-CREATE-Lab GitHub. These split metadata files are moved to the 'split' folder. The 'split_videos.py' can then be run to move the videos to their respective folders: 'train_set', 'validation_set', 'test_set'.
  4. **Run training script**: The 'train.py' script can be run by the following command: ``` python train.py ```
  5. **Run linear evaluation script**: The 'lineareval.py' script can be run by the following command: ``` python lineareval.py ```
  6. **Run baseline script**: The 'baseline.py' script can be run by the following command: ``` python baseline.py ```
  
