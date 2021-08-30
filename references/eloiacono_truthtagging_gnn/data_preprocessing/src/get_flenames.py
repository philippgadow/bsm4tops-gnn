import os

def create_filelist(base_dir, sample):

    dir2lookin = os.path.join(base_dir, sample)
    walk_output = list(os.walk(dir2lookin))

    print(dir2lookin)
    print(walk_output)

    with open(os.path.join('/eos/user/e/eloiacon/SWAN_projects/GNN/ATLAS_GNN/latest_samples/tt-with-gnn-boosted/data_preprocessing', sample.lower()+'_filelist.txt'), 'w') as txt:
        for i in range(0, len(walk_output)):
            for j in range(len(walk_output[i][2])):
                txt.write(os.path.join(walk_output[i][0], walk_output[i][2][j]))
                txt.write('\n')


base_dir = '/eos/user/e/eloiacon/SWAN_project/GNN/ATLAS_GNN/latest_samples/tt-with-gnn-boosted'
create_filelist(base_dir, '0L_33-05_a_CUT_D')
