# Distribution-Shifted-Dataset-Creator tool
Creates shifted-variants of a given dataset for simulating realistic/challenging distributional shifted environments (out-off distribution conditions) for benchmarking ML models' robustness ability in any given task.

📄 **Paper (TNNLS, accepted):** *GeNeX: Genetic Network eXperts framework for addressing Validation Overfitting*

![jsd_partition_skin(1)](https://github.com/user-attachments/assets/6c10adc3-1a9a-4a06-a505-4b24a1be115a)

## How it works:

In brief, it works via a clustering-based approach exploiting the Jensen–Shannon divergence (JSD) as the core partitioning measure/metric; JSD measures the relative entropy in information given two distributions.
The algorithm begins with partitioning the dataset into two subsets. The process continues iteratively by computing the centroids of the two sets and reassigning the instances closest to each one. This process ends up with two high-JSD-separated sets corresponding to the train vs test. The algorithm converges extremely fast.


## Input/Output data structure:

The given version is implemented for image data; but can be easily extended to any type of data.
Given a dataset in the following format:
```
data/
├── class1/
    ├── instance_1_1, instance_1_2, ...
├── class2/                 
    ├── instance_2_1, instance_2_2, ...
...
```
It will create a shifted variant with user-defined shift magnitude and spliting ratio: train/val vs shifted-test sets in the following format:
```
shifted_data/
├── train/
    ├── class1/
        ├── instance_1_1, instance_1_2, ...
    ├── class2/                 
        ├── instance_2_1, instance_2_2, ...
├── val/
    ...
├── test (OOD)/
    ...
```

## How to Run/Use

1. Open CONFIGS.py -->
```bash
USER_CONFIG = {

    "VISUALIZER": 1, # When terminates, it will create a vizual for inspecting the split; similar to the one above.
    # Put 0 for off.

    "SPLIT_RATIO" : 0.3, # The desired <train/val> to <test> sets spliting ratio.
    # E.g. this will produce a 30% <train/val> and 70% <test> ratio split from the initial unsplit data size.

    "SHIFT_MAGNITUDE" : 0.0, # Controls the JSD magnitude for varying OOD shift testing challenge.
    # Vary from 0 to 0.5 with decreasing shifting effect.
    # 0 brings the highest JSD (the most challenging and hardest generalization task).
    # Minimum JSD corresponds for 0.5 (almost similar to i.i.d sampling; easy generalization task).

    "ROOT_DIR" : "tiny data sample",  # Change this to your dataset root. 
    # Data must have the structure skeleton presented in "Input/Output data structure" of main GitHub page, 
    # with jpg images as instances.
    "OUTPUT_DIR" : "shifted_data", # Output path
    }
CACHE_PATH = USER_CONFIG["ROOT_DIR"].split()[0]+'_embeddings.npz'
# embedding saving / this saves the embeddings of imput dataset images.
# It is computed once. If you re-run, the module will load them, saving significant time.
```
read carefully the user config instructions inside and specify your desired configurations.

2. Then, just run:
```bash 
main.py
```
and will produce the output folder format described above.

3. Finally, we provide a tiny sample set in < tiny data sample > folder for debugging-running purposes before you want to actually apply into your datasets.


