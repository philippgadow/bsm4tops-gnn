import uproot
import numpy as np

# Input / Output helper functions

def getDataFrame(filename, treename='LHEF'):
    """Utility function providing pandas dataframe from ROOT input file."""
    with uproot.open(filename) as f:
        tree = f[treename]['Particle']
        df = tree.arrays(
                [
                    "Particle.PID", "Particle.Mother1", "Particle.Mother2",
                    "Particle.PT", "Particle.Eta", "Particle.Phi", "Particle.M"
                ],
                library="pd")
    return df


def cleanDataFrame(df):
    """Utility function to clean pandas dataframe and process information about resonance candidate."""
    # add column specifying whether the top quark originated from a resonance or not:
    # - 1: originates from resonance
    # - 0: originates from associated production
    df['resonance'] = df.apply(lambda row: 1 if row["Particle.Mother1"] == row["Particle.Mother2"] else 0, axis=1)

    # only keep top quarks in dataframe
    df = df[np.abs(df["Particle.PID"]) == 6]

    # remove columns which are no longer needed
    df = df.drop(["Particle.Mother1", "Particle.Mother2", "Particle.PID"], axis=1)
    return df


def augmentDataFrame(df):
    """Utility function to add additional variables to data frame (e.g. invariant mass or DeltaR)."""

    # TODO: add variables to dataframe, e.g. a useful variable could be the closest distance to another top quark

    return df