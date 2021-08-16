import hist
import vector
import uproot
import numpy as np
import mplhep as hep
import matplotlib.pyplot as plt
from argparse import ArgumentParser


def getArgumentParser():
    """Provide a parser which processes the command line arguments provided by the user.
    
    Returns:
        parser: ArgumentParser object which can provide command line arguments via
                parser.parse_arguments()
    """
    parser = ArgumentParser()
    parser.add_argument('inputFile')
    return parser


def getArrays(filename, treename='LHEF'):
    """Open a ROOT file and extract the four top quarks inside the file as numpy arrays.   
    
    Args:
        filename (str): path to ROOT file
        treename (str, optional): name of TTree inside ROOT file with LHE event record
    
    Returns:
        dict: Nested dictionary with the four top quarks in the final state:
              - keys range from 'top1' to 'top4'
              - the four momentum of the first top quark can be accessed via dict['top1']['p4']
              - the information whether the first top quark originates from the resonance or not
                can be accessed via dict['top1']['resonance']   
    """

    with uproot.open(filename) as f:
        # retrieve ROOT TTree called "Particle" 
        # which contains simulated top quarks and other particles
        particles = f[treename]['Particle']

        # get all properties simulated particles in the events as numpy arrays
        # - status: internal status code of MadGraph
        #           -1: initial state particle
        #            2: decayed particle
        #            1: final state particle
        # - PID: PDG particle ID (see https://pdg.lbl.gov/2019/reviews/rpp2019-rev-monte-carlo-numbering.pdf)
        #            21: gluon
        #            6000055: Z' boson
        #            6, -6: top quark
        # - mother1/2: mother particle in decay chain
        #            assuming fixed order of particles in LHE event record:
        #            [gluon, gluon, resonance, top1, top2, top3, top4]
        #            the gluons have mother1 and mother2 "-1" because they are initial state particles
        #            the resonance has mother1 "0" and mother2 "1" because it is created from the two gluons
        #            the associated production top quarks have mother1 "0" and mother2 "1" because they are also created from the gluons
        #            the resonance top quarks have mother1 "2" and mother2 "2" because they both originate from the resonance
        p_status = particles['Particle.Status'].array()
        p_pid = particles['Particle.PID'].array()
        p_mother1 = particles['Particle.Mother1'].array()
        p_mother2 = particles['Particle.Mother2'].array()

        # get four-vector arrays of first top quark (the one at last position of the array)
        top1_p4 = vector.array({
            "pt": particles['Particle.PT'].array()[:,-1],
            "eta": particles['Particle.Eta'].array()[:,-1],
            "phi": particles['Particle.Phi'].array()[:,-1],
            "M": particles['Particle.M'].array()[:,-1]
        })
        # get array with boolean indicating whether the top quark originates from the resonance or not
        top1_resonance = (
            particles['Particle.Mother1'].array()[:,-1] == \
            particles['Particle.Mother2'].array()[:,-1]
        )

        # information for third top quark (second-to-last position in array)
        top2_p4 = vector.array({
            "pt": particles['Particle.PT'].array()[:,-2],
            "eta": particles['Particle.Eta'].array()[:,-2],
            "phi": particles['Particle.Phi'].array()[:,-2],
            "M": particles['Particle.M'].array()[:,-2]
        })
        top2_resonance = (
            particles['Particle.Mother1'].array()[:,-2] == \
            particles['Particle.Mother2'].array()[:,-2]
        )

        # information for second top quark (third-to-last position in array)
        top3_p4 = vector.array({
            "pt": particles['Particle.PT'].array()[:,-3],
            "eta": particles['Particle.Eta'].array()[:,-3],
            "phi": particles['Particle.Phi'].array()[:,-3],
            "M": particles['Particle.M'].array()[:,-3]
        })
        top3_resonance = (
            particles['Particle.Mother1'].array()[:,-3] == \
            particles['Particle.Mother2'].array()[:,-3]
        )

        # information for fourth top quark (fourth-to-last position in array)
        top4_p4 = vector.array({
            "pt": particles['Particle.PT'].array()[:,-4],
            "eta": particles['Particle.Eta'].array()[:,-4],
            "phi": particles['Particle.Phi'].array()[:,-4],
            "M": particles['Particle.M'].array()[:,-4]
        })
        top4_resonance = (
            particles['Particle.Mother1'].array()[:,-4] == \
            particles['Particle.Mother2'].array()[:,-4]
        )

    return {
        'top1': {
            "p4": top1_p4,
            "resonance": top1_resonance
        },
        'top2': {
            "p4": top2_p4,
            "resonance": top2_resonance
        },
        'top3': {
            "p4": top3_p4,
            "resonance": top3_resonance
        },
        'top4': {
            "p4": top4_p4,
            "resonance": top4_resonance
        },
    }


def plotEventVariables(tops_p4):
    """Take as input a dictionary
    
    Args:
        tops_p4 (list): List of four elements, containing the four vectors of the top quarks
    """

    # define histograms
    h_mass_resonance = hist.Hist.new.Reg(30, 0, 3200).Int64()
    h_mass_spectator = hist.Hist.new.Reg(30, 0, 3200).Int64()
    h_dRtt_resonance = hist.Hist.new.Reg(30, 0, 6).Int64()
    h_dRtt_spectator = hist.Hist.new.Reg(30, 0, 6).Int64()
    h_dist_resonance = hist.Hist.new.Reg(30, 0, 6).Int64()
    h_dist_spectator = hist.Hist.new.Reg(30, 0, 8).Int64()


    inv_mass_resonance = (tops_p4[0] + tops_p4[1]).mass
    inv_mass_spectator = (tops_p4[2] + tops_p4[3]).mass
    dRtt_resonance = tops_p4[0].deltaR(tops_p4[1])
    dRtt_spectator = tops_p4[2].deltaR(tops_p4[3])


    for i,j,k in zip(tops_p4[0].deltaR(tops_p4[1]),tops_p4[0].deltaR(tops_p4[2]),tops_p4[0].deltaR(tops_p4[3])):
        h_dist_resonance.fill(min(i,j,k))
    for i,j,k in zip(tops_p4[1].deltaR(tops_p4[0]),tops_p4[1].deltaR(tops_p4[2]),tops_p4[1].deltaR(tops_p4[3])):
        h_dist_resonance.fill(min(i,j,k))
    for i,j,k in zip(tops_p4[2].deltaR(tops_p4[0]),tops_p4[2].deltaR(tops_p4[1]),tops_p4[2].deltaR(tops_p4[3])):
        h_dist_spectator.fill(min(i,j,k))
    for i,j,k in zip(tops_p4[3].deltaR(tops_p4[0]),tops_p4[3].deltaR(tops_p4[1]),tops_p4[3].deltaR(tops_p4[2])):
        h_dist_spectator.fill(min(i,j,k))


    h_mass_resonance.fill(inv_mass_resonance)
    h_mass_spectator.fill(inv_mass_spectator)
    h_dRtt_resonance.fill(dRtt_resonance)
    h_dRtt_spectator.fill(dRtt_spectator)
    

    # style
    hep.style.use(hep.style.ATLAS)

    # plots
    fig1, ax1 = plt.subplots()
    h_mass_resonance.plot(ax=ax1, label=r'$m=1.5$ TeV, $c_{t}=1$, $\theta=0.8$')
    plt.xlabel(r'$m_{tt}$ (resonance) [GeV]')
    plt.ylabel('Events')
    plt.legend(loc=1)
    fig1.savefig('mass_resonance.png')
    plt.close()

    fig2, ax2 = plt.subplots()
    h_mass_spectator.plot(ax=ax2, label=r'$m=1.5$ TeV, $c_{t}=1$, $\theta=0.8$')
    plt.xlabel(r'$m_{tt}$ (spectator) [GeV]')
    fig2.savefig('mass_spectator.png')
    plt.close()

    fig3, ax3 = plt.subplots()
    h_dRtt_resonance.plot(ax=ax3)
    plt.xlabel(r'$dR_{tt}$ (resonance)')
    fig3.savefig('dR_resonance.png')
    plt.close()

    fig4, ax4 = plt.subplots()
    h_dRtt_spectator.plot(ax=ax4)
    plt.xlabel(r'$dR_{tt}$ (spectator)')
    fig4.savefig('dR_spectator.png')
    plt.close()

    fig5, ax5 = plt.subplots()
    h_dist_resonance.plot(ax=ax5)
    plt.xlabel(r'dist to nearest top (resonance)')
    fig5.savefig('dist_resonance.png')
    plt.close()

    fig6, ax6 = plt.subplots()
    h_dist_spectator.plot(ax=ax6)
    plt.xlabel(r'dist to nearest top (spectator)')
    fig6.savefig('dist_spectator.png')
    plt.close()
    


def plotTopVariables(tops_p4):
    """Plot kinematic variables (pt, eta, phi) of top quarks
    
    Args:
        tops_p4 (list): List of four elements, containing the four vectors of the top quarks
    """


    # define empty lists to hold the four histograms for the top quarks
    h_top_pt = []
    h_top_eta = []
    h_top_phi = []


    # loop over four top quarks and collect distributions of top quark kinematics in histograms
    for i, p4 in enumerate(tops_p4):
        # histograms for top quark four-vector (except for its mass)
        h_top_pt.append(hist.Hist.new.Reg(30, 0, 1000).Int64())
        h_top_eta.append(hist.Hist.new.Reg(30, -5, 5).Int64())
        h_top_phi.append(hist.Hist.new.Reg(30, -4, 4).Int64())

        h_top_pt[i].fill(tops_p4[i].pt)
        h_top_eta[i].fill(tops_p4[i].eta)
        h_top_phi[i].fill(tops_p4[i].phi)

    # plotting style
    hep.style.use(hep.style.ATLAS)

    # make plots
    fig, ax = plt.subplots()
    for i, h in enumerate(h_top_pt):
        h.plot(ax=ax, label='top quark #{i}'.format(i=i))
    plt.xlabel(r'$p_{T}$ [GeV]')
    plt.ylabel('Events')
    plt.legend(loc=1)
    fig.savefig('top_pt.png')
    plt.close()

    fig, ax = plt.subplots()
    for i, h in enumerate(h_top_eta):
        h.plot(ax=ax, label='top quark #{i}'.format(i=i))
    plt.xlabel(r'$\eta$')
    plt.ylabel('Events')
    plt.legend(loc=1)
    fig.savefig('top_eta.png')
    plt.close()

    fig, ax = plt.subplots()
    for i, h in enumerate(h_top_phi):
        h.plot(ax=ax, label='top quark #{i}'.format(i=i))
    plt.xlabel(r'$\phi$')
    plt.ylabel('Events')
    plt.legend(loc=1)
    fig.savefig('top_phi.png')
    plt.close()


def main():
    # get histograms from files
    args = getArgumentParser().parse_args()

    # get nested dictionary with final state top quarks
    data = getArrays(args.inputFile)

    # collect top quark four-momenta into a list
    tops_p4 = [
        data['top1']['p4'],
        data['top2']['p4'],
        data['top3']['p4'],
        data['top4']['p4']
    ]

    # make plots
    plotTopVariables(tops_p4)
    plotEventVariables(tops_p4)


if __name__ == '__main__':
    main()
