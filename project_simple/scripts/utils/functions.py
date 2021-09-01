import vector
import numpy as np

def compute_dR(new_df, node_features, edges_src, edges_dst):
    result = []
    v_array = vector.arr({
        "pt": new_df['Particle.PT'],
        "phi": new_df['Particle.Eta'],
        "eta": new_df['Particle.Phi'],
        "M": new_df['Particle.M'],
    })
    for i, j in zip(edges_src, edges_dst):
        result.append(v_array[i].deltaR(v_array[j]))
    return np.array(result)
