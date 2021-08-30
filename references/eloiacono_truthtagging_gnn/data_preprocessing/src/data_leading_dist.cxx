#include <iostream>
#include <assert.h>
#include <vector>
#include <stdlib.h>
#include "chrono"

// for root
#include "TFile.h"
#include <TTree.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>

#include <TH1F.h>
#include <TString.h>

#include <random>
#include <numeric>
#include <TLorentzVector.h>

using namespace std;




float calculate_deltaR(float eta1, float phi1, float eta2, float phi2){
    float deta = eta1-eta2;
    float dphi = phi1-phi2;
    if (dphi >= TMath::Pi()){
        dphi -= 2*TMath::Pi();
    } else if (dphi <= -TMath::Pi()){
        dphi += 2*TMath::Pi();
    }

    return TMath::Sqrt(deta*deta + dphi*dphi);
}



struct TH1F_custom {

    int numbins;
    double max, min;
    TString name, title;

    TH1F *hist_b, *hist_c, *hist_l;
    TH1F_custom(TString name, TString title, int numbins, double min, double max){
        hist_b = new TH1F(name + "_b", title + " (bottom)" , numbins, min, max);
        hist_c = new TH1F(name + "_c", title + " (charm)" , numbins, min, max);
        hist_l = new TH1F(name + "_l", title + " (light)" , numbins, min, max);
    }

    void Fill(double val, double weight, int flav){
        if (flav == 5){
            hist_b->Fill(val, weight);
        } else if (flav == 4){
            hist_c->Fill(val, weight);
        } else if (flav == 0){
            hist_l->Fill(val, weight);
        }
    }

    void Write(){
        hist_b->Write(); hist_c->Write(); hist_l->Write();
    }
};



int main(int argc, char* argv[]) {
    
    /*
     Creates histograms -
        mJ and leading jet pT
            - events with only one jet
            - events with atleast two jets
     
     Args:
        argv[1]: path to the root file generated with data_prep2pred
        argv[2]: parth to output file
     */
    
    
    
    // start the timer
    auto t1 = std::chrono::high_resolution_clock::now();

    TFile* data_file;
    TString output_filepath;

    if (argc==3){

        data_file = new TFile(TString(argv[1]));
        output_filepath = TString(argv[2]);
        
    } else {
        std::cout << "Error: Requires 2 arguments. " << argc-1 << " were given" << std::endl;
        std::cout << "Exiting..." << std::endl;
        return 0;
    }

    std::cout << "histograms will be saved at " << output_filepath << std::endl;



    // read the data file
    TTreeReader data_reader("Nominal", data_file);

    TTreeReaderValue<std::vector<float>> jet_pt(data_reader, "jet_pt");
    TTreeReaderValue<std::vector<float>> jet_eta(data_reader, "jet_eta");
    TTreeReaderValue<std::vector<float>> jet_phi(data_reader, "jet_phi");
    TTreeReaderValue<std::vector<int>> jet_truthflav(data_reader, "jet_truthflav");

    TTreeReaderValue<std::vector<int>> jet_tag(data_reader, "jet_tag_btag_DL1r");

    // Truth Hadron and Truth Parton id info
    TTreeReaderValue<std::vector<float>> bH_m1(data_reader, "bH_m1");
    TTreeReaderValue<std::vector<float>> bH_pT1(data_reader, "bH_pT1");
    TTreeReaderValue<std::vector<float>> bH_eta1(data_reader, "bH_eta1");
    TTreeReaderValue<std::vector<float>> bH_phi1(data_reader, "bH_phi1");
    TTreeReaderValue<std::vector<int>> bH_pdgid1(data_reader, "bH_pdgid1");
    
    TTreeReaderValue<double> mJ(data_reader, "mJ");
    TTreeReaderValue<double> mu_actual(data_reader, "mu_actual");



    // File to write to
    TFile* outputfile = TFile::Open(output_filepath,"recreate");

    TH1F_custom *mass_hist_single_jet = new TH1F_custom("mass_hist_single_jet", "mJ (sinlge jet events)", 23, 20, 250);
    TH1F_custom *pt_hist_single_jet   = new TH1F_custom("pt_hist_single_jet", "Jet pT (single jet events)", 23, 20, 250);
    TH1F_custom *eta_hist_single_jet  = new TH1F_custom("eta_hist_single_jet", "Jet eta (single jet events)", 23, -3.14, 3.14);
    TH1F_custom *phi_hist_single_jet  = new TH1F_custom("phi_hist_single_jet", "Jet phi (single jet events)", 23, -3.14, 3.14);
    TH1F_custom *mu_hist_single_jet   = new TH1F_custom("mu_hist_single_jet", "mu (sinlge jet events)", 23, 0, 250);

    TH1F_custom *Hm_hist_single_jet   = new TH1F_custom("Hm_hist_single_jet", "H m (single jet events)", 23, 0, 250);
    TH1F_custom *Hpt_hist_single_jet  = new TH1F_custom("Hpt_hist_single_jet", "H pT (single jet events)", 23, 0, 250);
    TH1F_custom *Heta_hist_single_jet = new TH1F_custom("Heta_hist_single_jet", "H eta (single jet events)", 23, -3.14, 3.14);
    TH1F_custom *Hphi_hist_single_jet = new TH1F_custom("Hphi_hist_single_jet", "H phi (single jet events)", 23, -3.14, 3.14);

    TH1F_custom *pdgid_hist_single_jet = new TH1F_custom("pdgid_hist_single_jet", "pdgid (single jet events)" , 23, -1, 22);
    

    
    TH1F_custom *mass_hist_atleast_two_jets = new TH1F_custom("mass_hist_atleast_two_jets", "mJ (atleast two jets events)", 23, 20, 250);
    TH1F_custom *pt_hist_atleast_two_jets   = new TH1F_custom("pt_hist_atleast_two_jets", "Jet pT (atleast two jets events)", 23, 20, 250);
    TH1F_custom *eta_hist_atleast_two_jets  = new TH1F_custom("eta_hist_atleast_two_jets", "Jet eta (atleast two jets events)", 23, -3.14, 3.14);
    TH1F_custom *phi_hist_atleast_two_jets  = new TH1F_custom("phi_hist_atleast_two_jets", "Jet phi (atleast two jets events)", 23, -3.14, 3.14);
    TH1F_custom *mu_hist_atleast_two_jets   = new TH1F_custom("mu_hist_atleast_two_jets", "mu (atleast two jets events)", 23, 0, 250);

    TH1F_custom *Hm_hist_atleast_two_jets   = new TH1F_custom("Hm_hist_atleast_two_jets", "H m (atleast two jet events)", 23, 0, 250);
    TH1F_custom *Hpt_hist_atleast_two_jets  = new TH1F_custom("Hpt_hist_atleast_two_jets", "H pT (atleast two jet events)", 23, 0, 250);
    TH1F_custom *Heta_hist_atleast_two_jets = new TH1F_custom("Heta_hist_atleast_two_jets", "H eta (atleast two jet events)", 23, -3.14, 3.14);
    TH1F_custom *Hphi_hist_atleast_two_jets = new TH1F_custom("Hphi_hist_atleast_two_jets", "H phi (atleast two jet events)", 23, -3.14, 3.14);

    TH1F_custom *pdgid_hist_atleast_two_jets = new TH1F_custom("pdgid_hist_atleast_two_jets", "pdgid (atleast two jet events)" , 23, -1, 22);

    
    // event loop
    int num = 0;
    while (data_reader.Next()){

        num ++;
        if((num+1) % 1000000 == 0){
            std::cout << "Done with " << (num+1)/1000000 << "M events" << std::endl;
        }
        
        int N = jet_pt->size();

        if (N == 1){
            mass_hist_single_jet->Fill(*mJ, 1, jet_truthflav->at(0));
            mu_hist_single_jet->Fill(*mu_actual, 1, jet_truthflav->at(0));
            pt_hist_single_jet->Fill(jet_pt->at(0)/1000., 1, jet_truthflav->at(0));
            eta_hist_single_jet->Fill(jet_eta->at(0), 1, jet_truthflav->at(0));
            phi_hist_single_jet->Fill(jet_phi->at(0), 1, jet_truthflav->at(0));
            
	    Hm_hist_single_jet->Fill(bH_m1->at(0), 1, jet_truthflav->at(0));
	    Hpt_hist_single_jet->Fill(bH_pT1->at(0), 1, jet_truthflav->at(0));
	    Heta_hist_single_jet->Fill(bH_eta1->at(0), 1, jet_truthflav->at(0));
	    Hphi_hist_single_jet->Fill(bH_phi1->at(0), 1, jet_truthflav->at(0));
            
            pdgid_hist_single_jet->Fill(bH_pdgid1->at(0), 1, jet_truthflav->at(0));
            
        } else {
            mass_hist_atleast_two_jets->Fill(*mJ, 1, jet_truthflav->at(0));
            mu_hist_atleast_two_jets->Fill(*mu_actual, 1, jet_truthflav->at(0));
            pt_hist_atleast_two_jets->Fill(jet_pt->at(0)/1000., 1, jet_truthflav->at(0));
            eta_hist_atleast_two_jets->Fill(jet_eta->at(0), 1, jet_truthflav->at(0));
            phi_hist_atleast_two_jets->Fill(jet_phi->at(0), 1, jet_truthflav->at(0));

	    Hm_hist_atleast_two_jets->Fill(bH_m1->at(0), 1, jet_truthflav->at(0));
	    Hpt_hist_atleast_two_jets->Fill(bH_pT1->at(0), 1, jet_truthflav->at(0));
	    Heta_hist_atleast_two_jets->Fill(bH_eta1->at(0), 1, jet_truthflav->at(0));
	    Hphi_hist_atleast_two_jets->Fill(bH_phi1->at(0), 1, jet_truthflav->at(0));

            pdgid_hist_atleast_two_jets->Fill(bH_pdgid1->at(0), 1, jet_truthflav->at(0));
        }
    } // end of event loop


    // write the histograms to the root file
    mass_hist_single_jet->Write();
    mu_hist_single_jet->Write();
    pt_hist_single_jet->Write();
    eta_hist_single_jet->Write();
    phi_hist_single_jet->Write();
    
    Hm_hist_single_jet->Write();
    Hpt_hist_single_jet->Write();
    Heta_hist_single_jet->Write();
    Hphi_hist_single_jet->Write();
    
    pdgid_hist_single_jet->Write();
    
    
    mass_hist_atleast_two_jets->Write();
    mu_hist_atleast_two_jets->Write();
    pt_hist_atleast_two_jets->Write();
    eta_hist_atleast_two_jets->Write();
    phi_hist_atleast_two_jets->Write();
    
    Hm_hist_atleast_two_jets->Write();
    Hpt_hist_atleast_two_jets->Write();
    Heta_hist_atleast_two_jets->Write();
    Hphi_hist_atleast_two_jets->Write();
    
    outputfile->Close();

    
    // execution time
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();

    int total_min = (int)(duration / 60000000);
    double seconds = ((double)(duration / 1000000.)) - (double(total_min) * 60.);
    int min       = total_min % 60;
    int hours     = total_min / 60;

    std::cout << std::endl;
    std::cout << "Time taken: " << hours << " hrs " << min << " min "
              << seconds << " seconds " << std::endl;
    
    return 0;
}
