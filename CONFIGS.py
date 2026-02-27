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