# CAWL: Classification of AIS data with Weak Labelers

## Project Structure

```
cawl/
├── cawl/                    # Main package
│   ├── data/               # Data processing modules
│   │   ├── downloader.py   # AIS data downloader
│   │   ├── trajectory_regression_dataset.py  # Dataset for trajectory analysis
│   │   └── gp_kernel_dataset.py  # GP-specific dataset processing
│   ├── models/             # Machine learning models
│   │   ├── gp/             # Gaussian Process models (from Daniel)
│   │   │   └── multioutput_rbf_linear_gp.py
│   │   ├── tree/           # Decision tree models
│   │   │   └── decision_tree_model.py
│   │   └── wl/             # Weak labeling models (from Rattan)
│   │       └── wl.py
│   └── scripts/            # Utility scripts
│       └── ais_download.py # CLI for downloading AIS data
├── notebooks/
│   └── small_decision_trees.ipynb # Notebook pulling all of the code together
├── data/                   # Raw and processed data storage
├── models/                 # Saved model outputs
```
