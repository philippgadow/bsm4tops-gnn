## Identification of top quarks originating from a heavy resonance

In this project we will design and test various classifiers which can be used to identify the top quarks which originate from the decay of a heavy resonance.


### Setup

We rely on certain python packages which greatly simplify the analysis of ROOT files and provide tools for machine learning.
Therefore, we need to download and setup these packages. To make sure we do this in a clean environment, we resort to using a "virtual environment".

All is taken care of in the setup script.

Please follow these instructions before starting with the scripts:

```
cd plotting
source setup.sh

```


### Running the script

To run the script, first make sure that the virtual environment is active by sourcing the `setup.sh` script and execute:

```
python scripts/bsm4tops.py ../data/simple/unweighted_events.root

```

This will just run the plotting routine and provide an overview plot in `plots/overview.png`.


For running the script with evaluation of the simple classifier, execute:

```
python scripts/bsm4tops.py --simple ../data/simple/unweighted_events.root

```

For running the script with training and evaluation of a [k-nearest-neighbours (k-NN)]() classifier, execute:

```
python scripts/bsm4tops.py --bdt ../data/simple/unweighted_events.root

```

For running the script with training and evaluation of a [Boosted Decision Tree (BDT)]() classifier, execute:

```
python scripts/bsm4tops.py --bdt ../data/simple/unweighted_events.root

```

### Development notes

The script which you can take as the basis for your studies is located in 

```
scripts/bsm4tops.py
```

It also makes use of some utility functions which are provided in 

- `scripts/utils/io.py`
- `scripts/utils/plotting.py`

