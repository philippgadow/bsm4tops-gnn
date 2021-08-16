import uproot
import vector
import numpy as np
import pandas as pd


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
#    """Utility function to add additional variables to data frame (e.g. invariant mass or DeltaR)."""
#    df = df.assign(closest_distance=0.0)
#    df = df.assign(invariant_mass=0.0)
#    for index in range(0,20000):
#        top = [vector.obj(pt=df.xs(index)['Particle.PT'][3],phi=df.xs(index)['Particle.Phi'][3],eta=df.xs(index)['Particle.Eta'][3],mass=df.xs(index)['Particle.M'][3]),
#                vector.obj(pt=df.xs(index)['Particle.PT'][4],phi=df.xs(index)['Particle.Phi'][4],eta=df.xs(index)['Particle.Eta'][4],mass=df.xs(index)['Particle.M'][4]),
#                vector.obj(pt=df.xs(index)['Particle.PT'][5],phi=df.xs(index)['Particle.Phi'][5],eta=df.xs(index)['Particle.Eta'][5],mass=df.xs(index)['Particle.M'][5]),
#                vector.obj(pt=df.xs(index)['Particle.PT'][6],phi=df.xs(index)['Particle.Phi'][6],eta=df.xs(index)['Particle.Eta'][6],mass=df.xs(index)['Particle.M'][6])]
#        df.at[(index,3),'closest_distance'] = min(top[0].deltaR(top[1]),top[0].deltaR(top[2]),top[0].deltaR(top[3]))
#        df.at[(index,3),'invariant_mass'] = max((top[0] + top[1]).mass,(top[0] + top[2]).mass,(top[0] + top[3]).mass)
#        df.at[(index,4),'closest_distance'] = min(top[1].deltaR(top[0]),top[1].deltaR(top[2]),top[1].deltaR(top[3]))
#        df.at[(index,4),'invariant_mass'] = max((top[1] + top[0]).mass,(top[1] + top[2]).mass,(top[1] + top[3]).mass)
#        df.at[(index,5),'closest_distance'] = min(top[2].deltaR(top[0]),top[2].deltaR(top[1]),top[2].deltaR(top[3]))
#        df.at[(index,5),'invariant_mass'] = max((top[2] + top[0]).mass,(top[2] + top[1]).mass,(top[2] + top[3]).mass)
#        df.at[(index,6),'closest_distance'] = min(top[3].deltaR(top[0]),top[3].deltaR(top[1]),top[3].deltaR(top[2]))
#        df.at[(index,6),'invariant_mass'] = max((top[3] + top[0]).mass,(top[3] + top[1]).mass,(top[3] + top[2]).mass)
#        if index % 1000 == 0:
#            print(index)
#    print(df)
    return df
