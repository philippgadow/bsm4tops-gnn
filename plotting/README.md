## Exploring the dataset for the simple problem formulation

We will be using python to investigate the dataset. The simulated events are provided as an LHE file which is converted to a ROOT file.
LHE (Les Houches Event) is a standard format for the output from parton-level event generators.

For the analysis and plotting routines we exclusively rely on python tools.

More specifically:

- https://github.com/scikit-hep/uproot4
- https://github.com/scikit-hep/vector
- https://github.com/scikit-hep/hist


### Setup

We rely on certain python packages which greatly simplify the analysis of ROOT files and cater to the needs of experimental high energy physicists.
Therefore, we need to download and setup these packages. To make sure we do this in a clean environment, we resort to using a "virtual environment".

All is taken care of in the setup script.

Please follow these instructions before starting with the scripts:

```
cd plotting
source setup.sh
```


### Making plots

Now is the time where you can make your first plots!

There is already a script provided for you in `scripts`.

You can execute it as follows (assuming you have sourced the `setup.sh` script already):

```
python scripts/bsm4tops_plot.py ../data/simple/unweighted_events.root
```

As a result, you get a few `png` files with plots of the top quark kinematic properties.

Now it is your turn! Open the script in an editor and create the following set of plots:

- the invariant mass distribution for the top quarks from the associated production
- the DeltaR distrance between the top quarks from the resonance decay
- the DeltaR distrance between the top quarks from the associated production

Hints:

- have a look at the definition of the distance measure DeltaR and the LHC coordinate system
- look at the methods which the vector package provides to you (see [here](https://vector.readthedocs.io/en/latest/usage/intro.html#Vector-methods), you are looking for something which computes `deltaR`).
