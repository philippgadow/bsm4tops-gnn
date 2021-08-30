#include <iostream>
#include <fstream>
#include <iomanip>
#include <utility>
#include <vector>
#include <chrono>
#include <algorithm>
#include <random>

#include "TROOT.h"
#include "TSystem.h"
#include "TCanvas.h"
#include "TApplication.h"

#include "TString.h"
#include "TMath.h"

#include "TH3.h"
#include "TH2.h"
#include "TH1.h"
#include "TSystem.h"
#include "TImage.h"
#include "THStack.h"
#include "TLegend.h"
#include "TPaveText.h"
#include "TClonesArray.h"
#include "TLorentzVector.h"
#include "TVector3.h"

#include "Reader.C"

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





int main(int argc, char *argv[]){
    
    /*
     Shuffles the data and returns the train and eval root files
     means_std are calculated for only the selected events
     
     Argv:
        argv[1]: path to root file or the text file containing the path to the root files
        argv[2]: map path
        argv[3]: path to the train file
        argv[4]: path to the eval file
     */
    
    
    // start the timer
    auto t1 = std::chrono::high_resolution_clock::now();

    TChain* chain = new TChain("Nominal");
    TString output_filepath_train, output_filepath_eval, map_path, meanstd_filepath;

    if (argc==5){

        TString input_filepath  = TString(argv[1]);
        TString input_filetype = TString(input_filepath(input_filepath.Length()-4, input_filepath.Length()));

        if (input_filetype == TString(".txt")){
            std::ifstream file(input_filepath);
            std::string str;
            
            while (std::getline(file, str)) {
                chain->Add(TString(str));
            }
        } else {
            std::cout << "Following file will be used -" << std::endl;
            std::cout << input_filepath << std::endl;
            chain->Add(input_filepath);
        }
        
        map_path = TString(argv[2]);
        output_filepath_train = TString(argv[3]);
        output_filepath_eval = TString(argv[4]);
        meanstd_filepath = TString(output_filepath_train(0,output_filepath_train.Length()-5)).Append(TString("_meanstd.json"));

    } else {
        std::cout << "Error: Requires exactly 4 arguments. " << argc-1 << " were given" << std::endl;
        std::cout << "Exiting..." << std::endl;
        return 0;
    }
    std::cout << "train will be saved at " << output_filepath_train << std::endl;
    std::cout << "eval will be saved at " << output_filepath_eval << std::endl;


    
    


    Reader* reader = new Reader(chain);
    chain->SetBranchStatus("*",0);

    // Turn on the braches needed
    chain->SetBranchStatus("jetpf_pt", 1 );
    chain->SetBranchStatus("jetpf_eta", 1 );
    chain->SetBranchStatus("jetpf_phi", 1 );
    chain->SetBranchStatus("jetpf_truthflav", 1 );
    chain->SetBranchStatus("ActualMu", 1 );
    chain->SetBranchStatus("MET", 1 );

    // Truth Hadron and Truth Parton id info
    chain->SetBranchStatus("bH_m1", 1 );
    chain->SetBranchStatus("bH_pT1", 1 );
    chain->SetBranchStatus("bH_eta1", 1 );
    chain->SetBranchStatus("bH_phi1", 1 );
    chain->SetBranchStatus("bH_pdgid1", 1 );

    // DL1r info
    chain->SetBranchStatus("jetpf_DL1r_pb", 1 );
    chain->SetBranchStatus("jetpf_DL1r_pc", 1 );
    chain->SetBranchStatus("jetpf_DL1r_pu", 1 );

    // for event selection
    chain->SetBranchStatus("mBB", 1 );

    chain->SetBranchStatus("EventWeight", 1 );
    chain->SetBranchStatus("BTagSF", 1 );

    int num_entries = chain->GetEntries();
    std::cout << "total events: " << num_entries << std::endl;



    

    //*************************************************//
    // Find the means and stds for the selected events //
    //*************************************************//

    std::cout << "Calculating means and stds..." << std::endl;

    int jet_count = 0, dr_comb_count = 0;
    double mean_pt=0, mean_phi=0, mean_eta=0, mean_mu_actual=0, mean_dR=0;

    // Truth Hadron info
    int bH_count=0, cH_count=0;
    double
    mean_bH_m=0, mean_bH_pt=0, mean_bH_eta=0, mean_bH_phi=0,
    mean_cH_m=0, mean_cH_pt=0, mean_cH_eta=0, mean_cH_phi=0;

    double mean_MET=0, mean_mBB=0;

    for (int num = 0; num < num_entries; ++num){

        chain->GetEntry(num);
        reader->GetEntry(num);

        // skip if not a boosted event
        if (reader->mBB <= 0){
            continue;
        }

        // skip empty events
        int njets = reader->jetpf_pt->size();
        if (njets == 0){
            continue;
        }
        
        // skip single jet events
        if (njets == 1){
            continue;
        }
        
        // skip high pT (extreme) event
        if (reader->jetpf_pt->at(0) > 1E6){
            continue;
        }

        for (int i=0; i<njets; ++i){
            jet_count++;
            mean_pt        += reader->jetpf_pt->at(i);
            mean_phi       += reader->jetpf_phi->at(i);
            mean_eta       += reader->jetpf_eta->at(i);
            mean_mu_actual += reader->ActualMu;

            for (int j=0; j<njets; ++j){
                if(i==j){ continue; }
                dr_comb_count++;
                mean_dR += calculate_deltaR(reader->jetpf_eta->at(i), reader->jetpf_phi->at(i), reader->jetpf_eta->at(j), reader->jetpf_phi->at(j));
		 }

            // Truth Hadron info
            // if (reader->jetpf_truthflav->at(i) == 5){
            //     bH_count++;
            //     mean_bH_m   += reader->bH_m1->at(i);
            //     mean_bH_pt  += reader->bH_pT1->at(i);
            //     mean_bH_eta += reader->bH_eta1->at(i);
            //     mean_bH_phi += reader->bH_phi1->at(i);
            // } else if (reader->jetpf_truthflav->at(i) == 4){
            //     cH_count++;
            //     mean_cH_m   += reader->bH_m1->at(i);
            //     mean_cH_pt  += reader->bH_pT1->at(i);
            //     mean_cH_eta += reader->bH_eta1->at(i);
            //     mean_cH_phi += reader->bH_phi1->at(i);
            // }
            
	    // mean_mBB  += reader->mBB;
            mean_MET += reader->MET;
        }
    }

    mean_pt = mean_pt/jet_count; mean_phi = mean_phi/jet_count;
    mean_eta = mean_eta/jet_count; mean_mu_actual = mean_mu_actual/jet_count;
    mean_dR = mean_dR/dr_comb_count;

    // Truth Hadron info
    //    mean_bH_m=mean_bH_m/bH_count; mean_bH_pt=mean_bH_pt/bH_count; mean_bH_eta=mean_bH_eta/bH_count; mean_bH_phi=mean_bH_phi/bH_count;
    // mean_cH_m=mean_cH_m/cH_count; mean_cH_pt=mean_cH_pt/cH_count; mean_cH_eta=mean_cH_eta/cH_count; mean_cH_phi=mean_cH_phi/cH_count;

    // mean_mBB = mean_mBB/jet_count;
    mean_MET = mean_MET/jet_count;


    double std_pt=0, std_phi=0, std_eta=0, std_mu_actual=0, std_dR=0;

    // Truth Hadron info
    double
    std_bH_m=0, std_bH_pt=0, std_bH_eta=0, std_bH_phi=0,
    std_cH_m=0, std_cH_pt=0, std_cH_eta=0, std_cH_phi=0;

    double std_mBB=0, std_MET=0;

    for (int num = 0; num < num_entries; ++num){

        chain->GetEntry(num);
        reader->GetEntry(num);

        // skip if not a boosted event
        if (reader->mBB <= 0){
            continue;
        }

        // skip empty events
        int njets = reader->jetpf_pt->size();
        if (njets == 0){
            continue;
        }
        
        // skip single jet events
        if (njets == 1){
            continue;
        }
        
        // skip high pT (extreme) event
        if (reader->jetpf_pt->at(0) > 1E6){
            continue;
        }
        
        for (int i=0; i<njets; ++i){
            std_pt        += TMath::Power((reader->jetpf_pt->at(i) - mean_pt), 2);
            std_phi       += TMath::Power((reader->jetpf_phi->at(i) - mean_phi), 2);
            std_eta       += TMath::Power((reader->jetpf_eta->at(i) - mean_eta), 2);
            std_mu_actual += TMath::Power((reader->ActualMu - mean_mu_actual), 2);

            for (int j=0; j<njets; ++j){
                if(i==j){ continue; }
                double dR = calculate_deltaR(reader->jetpf_eta->at(i), reader->jetpf_phi->at(i), reader->jetpf_eta->at(j), reader->jetpf_phi->at(j));
                std_dR += TMath::Power((dR - mean_dR), 2);
            }

            // Truth Hadron info
            // if (reader->jetpf_truthflav->at(i) == 5){
            //     std_bH_m += TMath::Power((reader->bH_m1->at(i) - mean_bH_m), 2);
            //     std_bH_pt  += TMath::Power((reader->bH_pT1->at(i) - mean_bH_pt), 2);
            //     std_bH_eta += TMath::Power((reader->bH_eta1->at(i) - mean_bH_eta), 2);
            //     std_bH_phi += TMath::Power((reader->bH_phi1->at(i) - mean_bH_phi), 2);
            // } else if (reader->jetpf_truthflav->at(i) == 4){
            //     std_cH_m += TMath::Power((reader->bH_m1->at(i) - mean_cH_m), 2);
            //     std_cH_pt  += TMath::Power((reader->bH_pT1->at(i) - mean_cH_pt), 2);
            //     std_cH_eta += TMath::Power((reader->bH_eta1->at(i) - mean_cH_eta), 2);
            //     std_cH_phi += TMath::Power((reader->bH_phi1->at(i) - mean_cH_phi), 2);
            // }
            
	    //      std_mBB  += TMath::Power((reader->mBB - mean_mBB), 2);
            std_MET += TMath::Power((reader->MET - mean_MET), 2);
        }
    }

    std_pt = TMath::Sqrt(std_pt/jet_count); std_phi = TMath::Sqrt(std_phi/jet_count);
    std_eta = TMath::Sqrt(std_eta/jet_count); std_mu_actual = TMath::Sqrt(std_mu_actual/jet_count);
    std_dR = TMath::Sqrt(std_dR/dr_comb_count);

    // Truth Hadron info
    // std_bH_pt  = TMath::Sqrt(std_bH_pt/bH_count);  std_bH_eta = TMath::Sqrt(std_bH_eta/bH_count);
    // std_bH_phi = TMath::Sqrt(std_bH_phi/bH_count); std_bH_m = TMath::Sqrt(std_bH_m/bH_count);
    // std_cH_pt  = TMath::Sqrt(std_cH_pt/cH_count);  std_cH_eta = TMath::Sqrt(std_cH_eta/cH_count);
    // std_cH_phi = TMath::Sqrt(std_cH_phi/cH_count); std_cH_m = TMath::Sqrt(std_cH_m/cH_count);

    //  std_mBB  = TMath::Sqrt(std_mBB/jet_count);
    std_MET = TMath::Sqrt(std_MET/jet_count);

    // write the mean and stds to a json file
    std::cout << "Creating " << meanstd_filepath << "..." << endl;
    const int num_var = 5+0+1;
    const char *var_names[num_var] = {
        "jet_pt", "jet_eta", "jet_phi", "mu_actual", "dR",

        // Truth Hadron info
        // "bH_m", "bH_pt", "bH_eta", "bH_phi",
        // "cH_m", "cH_pt", "cH_eta", "cH_phi",
        
        "MET" //, "mBB"
    };
    double means[num_var] = {
        mean_pt, mean_eta, mean_phi, mean_mu_actual, mean_dR,

        // Truth Hadron info
        // mean_bH_m, mean_bH_pt, mean_bH_eta, mean_bH_phi,
        // mean_cH_m, mean_cH_pt, mean_cH_eta, mean_cH_phi,
        
        mean_MET //, mean_mBB
    };
    double stds[num_var]  = {
        std_pt, std_eta, std_phi, std_mu_actual, std_dR,

        // Truth Hadron info
        // std_bH_m, std_bH_pt, std_bH_eta, std_bH_phi,
        // std_cH_m, std_cH_pt, std_cH_eta, std_cH_phi,
        
        std_MET //, std_mBB
    };

    ofstream myfile;
    myfile.open (meanstd_filepath);

    myfile << "{" << endl;
    for (int i=0; i<num_var; ++i){
        myfile <<"\t\""<< var_names[i] <<"\""<< ": {" << endl;
        myfile << std::fixed <<"\t\t\"mean\": \"" << means[i] << "\"," << endl;
        myfile << std::fixed << "\t\t\"std\": \""  << stds[i] << "\"" << endl;
        if (i==num_var-1){
            myfile << "\t}" << endl;
        } else {
            myfile << "\t}," << endl;
        }
    }
    myfile << "}" << endl;
    myfile.close();





    // ************************************************//
    // select the events, and store and in _br vectors //
    // ************************************************//

    std::vector<std::vector<float>> jetpf_pt_br, jetpf_eta_br, jetpf_phi_br;
    std::vector<std::vector<int>> jetpf_truthflav_br;
    std::vector<float> ActualMu_br, MET_br;
    std::vector<double> mBB_br;

    // for truth hadron adn parton id info
    std::vector<std::vector<float>> bH_m1_br, bH_pT1_br, bH_eta1_br, bH_phi1_br;
    std::vector<std::vector<int>> bH_pdgid1_br;

    // for DL1r info
    std::vector<std::vector<double>> jetpf_DL1r_pb_br, jetpf_DL1r_pc_br, jetpf_DL1r_pu_br;

    std::vector<float> EventWeight_br;

    // event loop for making the _br vectors
    std::cout << "reading the data, and temporarily storing the selected events..." << std::endl;
    int num_selcted_events = 0, num_empty_events = 0;
    for (int num = 0; num < num_entries; ++num){

        if((num+1) % int(1e6) == 0){
            std::cout << "Done with " << (num+1)/int(1e6) << "M events" <<std::endl;
        }

        chain->GetEntry(num);
        reader->GetEntry(num);

        // skip if not a boosted event
        if (reader->mBB <= 0){
            continue;
        }

        // skip empty events
        int njets = reader->jetpf_pt->size();
        if (njets == 0){
            num_empty_events++;
            continue;
        }
        
        // skip single jet events
        if (njets == 1){
            continue;
        }

        // skip high pT (extreme) event
        if (reader->jetpf_pt->at(0) > 1E6){
            continue;
        }

        num_selcted_events++;

        // create the tmp vectors
        std::vector<float> jetpf_pt, jetpf_eta, jetpf_phi;
        std::vector<int> jetpf_truthflav;

        // for truth hadron adn parton id info
        std::vector<float> bH_m1, bH_pT1, bH_eta1, bH_phi1;
        std::vector<int> bH_pdgid1;

        // for DL1r info
        std::vector<double> jetpf_DL1r_pb, jetpf_DL1r_pc, jetpf_DL1r_pu;


        for (int i=0; i<njets; ++i){

            jetpf_pt.push_back(reader->jetpf_pt->at(i));
            jetpf_eta.push_back(reader->jetpf_eta->at(i));
            jetpf_phi.push_back(reader->jetpf_phi->at(i));
            jetpf_truthflav.push_back(reader->jetpf_truthflav->at(i));

            // if ((reader->jetpf_truthflav->at(i) == 5) || (reader->jetpf_truthflav->at(i) == 4)){
            //     bH_m1.push_back(reader->bH_m1->at(i));
            //     bH_pT1.push_back(reader->bH_pT1->at(i));
            //     bH_eta1.push_back(reader->bH_eta1->at(i));
            //     bH_phi1.push_back(reader->bH_phi1->at(i));
            // } else {
            //     bH_m1.push_back(0);
            //     bH_pT1.push_back(0);
            //     bH_eta1.push_back(0);
            //     bH_phi1.push_back(0);
            // }

            // bH_pdgid1.push_back(reader->bH_pdgid1->at(i));

            jetpf_DL1r_pb.push_back(reader->jetpf_DL1r_pb->at(i));
            jetpf_DL1r_pc.push_back(reader->jetpf_DL1r_pc->at(i));
            jetpf_DL1r_pu.push_back(reader->jetpf_DL1r_pu->at(i));

        } // jet loop ends

        float ActualMu = reader->ActualMu;
        float MET = reader->MET;
        double mBB = reader->mBB;
        float EventWeight = reader->EventWeight / reader->BTagSF;

        // fill the branch vectors
        jetpf_pt_br.push_back(jetpf_pt); jetpf_eta_br.push_back(jetpf_eta); jetpf_phi_br.push_back(jetpf_phi);
        jetpf_truthflav_br.push_back(jetpf_truthflav);
        ActualMu_br.push_back(ActualMu); mBB_br.push_back(mBB); EventWeight_br.push_back(EventWeight);
        MET_br.push_back(MET);

	//        bH_m1_br.push_back(bH_m1); bH_pT1_br.push_back(bH_pT1); bH_eta1_br.push_back(bH_eta1); bH_phi1_br.push_back(bH_phi1);
	// bH_pdgid1_br.push_back(bH_pdgid1);

        jetpf_DL1r_pb_br.push_back(jetpf_DL1r_pb); jetpf_DL1r_pc_br.push_back(jetpf_DL1r_pc); jetpf_DL1r_pu_br.push_back(jetpf_DL1r_pu);

    } // event loop ends

    std::cout << "Total boosted events: " << num_selcted_events << std::endl;








    //*********************//
    // Shuffle the entries //
    //*********************//

    // make the vector with all the indices to events
    std::vector<int> indices;
    for (int i=0; i<num_selcted_events; i++){
        indices.push_back(i);
    }

    // shuffle the indices
    auto rd = std::random_device {};
    auto rng = std::default_random_engine { rd() };
    std::shuffle(std::begin(indices), std::end(indices), rng);






    //**************************//
    // Make the evaluation file //
    //**************************//

    // Histogram based efficiency
    TFile* map_file = new TFile(map_path);
    TH2F* bottom_eff = (TH2F*)map_file->Get("bottom_70eff");
    TH2F* charm_eff  = (TH2F*)map_file->Get("charm_70eff");
    TH2F* light_eff  = (TH2F*)map_file->Get("light_70eff");
    TH2F* tau_eff    = (TH2F*)map_file->Get("tau_70eff");

    // All histograms have same binning
    int n_bins_pt = bottom_eff->GetXaxis()->GetNbins();
    int n_bins_eta = bottom_eff->GetYaxis()->GetNbins();

    double max_pt  = bottom_eff->GetXaxis()->GetBinLowEdge(n_bins_pt) + bottom_eff->GetXaxis()->GetBinWidth(n_bins_pt);
    double max_eta = bottom_eff->GetYaxis()->GetBinLowEdge(n_bins_eta) + bottom_eff->GetYaxis()->GetBinWidth(n_bins_eta);

    double min_pt  = bottom_eff->GetXaxis()->GetBinLowEdge(1);
    double min_eta = bottom_eff->GetYaxis()->GetBinLowEdge(1);


    // create the root file
    TFile* outputfile_eval = TFile::Open(output_filepath_eval, "recreate");
    TTree* newtree_eval = new TTree("Nominal","Nominal");

    std::vector<float> jet_pt, jet_eta, jet_phi, jet_e, custom_eff;
    std::vector<int> jet_truthflav, jet_tag_btag_DL1r;
    double mu_actual, mBB;
    float MET;
    float EventWeight;
    
    std::vector<float> H_m1, H_pT1, H_eta1, H_phi1;

    newtree_eval->Branch("jet_pt", &jet_pt);
    newtree_eval->Branch("jet_eta", &jet_eta);
    newtree_eval->Branch("jet_phi", &jet_phi);
    newtree_eval->Branch("jet_e", &jet_e);

    newtree_eval->Branch("jet_truthflav", &jet_truthflav);
    newtree_eval->Branch("mu_actual", &mu_actual);
    newtree_eval->Branch("mBB", &mBB);
    newtree_eval->Branch("MET", &MET);
    newtree_eval->Branch("EventWeight", &EventWeight);

    newtree_eval->Branch("jet_tag_btag_DL1r", &jet_tag_btag_DL1r);
    newtree_eval->Branch("custom_eff", &custom_eff);
    
    newtree_eval->Branch("H_m1", &H_m1);
    newtree_eval->Branch("H_pT1", &H_pT1);
    newtree_eval->Branch("H_eta1", &H_eta1);
    newtree_eval->Branch("H_phi1", &H_phi1);

    // pdg parton
    std::vector<int> pdg_parton;
    newtree_eval->Branch("pdg_parton", &pdg_parton);


    double f = 0.018, threshold_DL1r = 3.515; // 70% VHbb

    // event loop for making tree with shuffled entries
    std::cout << "making the eval data file..." << std::endl;
    for (int num_unshuffled = 0; num_unshuffled < num_selcted_events; ++num_unshuffled){

        if((num_unshuffled+1) % int(1e6) == 0){
            std::cout << "Done with " << (num_unshuffled+1)/int(1e6) << "M events" <<std::endl;
        }

        int num = indices[num_unshuffled];


        // clear stuff here
        jet_pt.clear(); jet_eta.clear(); jet_phi.clear(); jet_truthflav.clear();
        jet_e.clear();
        jet_tag_btag_DL1r.clear(); custom_eff.clear();
        pdg_parton.clear();

        H_m1.clear(); H_pT1.clear(); H_eta1.clear(); H_phi1.clear();

        int njets = jetpf_pt_br[num].size();
        for (int i=0; i<njets; ++i){

            jet_pt.push_back(jetpf_pt_br[num][i]);
            jet_eta.push_back(jetpf_eta_br[num][i]);
            jet_phi.push_back(jetpf_phi_br[num][i]);

            jet_truthflav.push_back(jetpf_truthflav_br[num][i]);

            // jet_tag VR
            double DL1r = TMath::Log(jetpf_DL1r_pb_br[num][i] / (jetpf_DL1r_pc_br[num][i]*f + (1-f)*jetpf_DL1r_pu_br[num][i]));
            if (DL1r > threshold_DL1r){
                jet_tag_btag_DL1r.push_back(1);
            } else {
                jet_tag_btag_DL1r.push_back(0);
            }

            // custom map eff
            double eff=0; /* what's a good default efficiency value?? */

            // fixing the overflow/underflow issue
            double pt_value  = jetpf_pt_br[num][i];
            double eta_value = TMath::Abs(jetpf_eta_br[num][i]);

            if (pt_value >= max_pt){
                pt_value = max_pt - 0.0001;
            } else if (pt_value <= min_pt){
                pt_value = min_pt + 0.0001;
            }

            if (eta_value >= max_eta){
                eta_value = max_eta - 0.0001;
            } else if (eta_value <= min_eta){
                eta_value = min_eta + 0.0001;
            }

            switch (jetpf_truthflav_br[num][i]){
                case 5: {
                    int bin = bottom_eff->FindBin(pt_value, eta_value);
                    eff = bottom_eff->GetBinContent(bin);
                    break;
                }
                case 4: {
                    int bin = charm_eff->FindBin(pt_value, eta_value);
                    eff = charm_eff->GetBinContent(bin);
                    break;
                }
                case 0: {
                    int bin = light_eff->FindBin(pt_value, eta_value);
                    eff = light_eff->GetBinContent(bin);
                    break;
                }
                case 15: {
                    int bin = tau_eff->FindBin(pt_value, eta_value);
                    eff = tau_eff->GetBinContent(bin);
                    break;
                }
            }
            custom_eff.push_back(eff);
            
            // H_m1.push_back(bH_m1_br[num][i]);
            // H_pT1.push_back(bH_pT1_br[num][i]);
            // H_eta1.push_back(bH_eta1_br[num][i]);
            // H_phi1.push_back(bH_phi1_br[num][i]);


            // pdg info
        //     pdg_parton.push_back(bH_pdgid1_br[num][i]);
         }

        mu_actual = ActualMu_br[num];
        MET = MET_br[num];
	//  mBB = mBB_br[num];
        EventWeight = EventWeight_br[num];

        //record current vector as entry in tree
        newtree_eval->Fill();

    } // event loop ends

    newtree_eval->Write();
    outputfile_eval->Close();






   //*********************//
   // Make the train file //
   //*********************//

    // Create the output file to save to, and the tree for that
    TFile* outputfile_train = TFile::Open(output_filepath_train,"recreate");
    TTree* newtree_train = new TTree("Nominal","Nominal");

    std::vector<float> features, deltaR;
    std::vector<int> jet_flavindex;

    newtree_train->Branch("features", &features);
    newtree_train->Branch("jet_truthflav", &jet_truthflav);
    newtree_train->Branch("deltaR", &deltaR);
    newtree_train->Branch("jet_tag_btag_DL1r", &jet_tag_btag_DL1r);
    newtree_train->Branch("jet_flavindex", &jet_flavindex);


    // event loop for making tree with shuffled entries
    std::cout << "making the train data file..." << std::endl;
    for (int num_unshuffled = 0; num_unshuffled < num_selcted_events; ++num_unshuffled){

        if((num_unshuffled+1) % int(1e6) == 0){
            std::cout << "Done with " << (num_unshuffled+1)/int(1e6) << "M events" <<std::endl;
        }

        int num = indices[num_unshuffled];

        // clear stuff here
        features.clear(); jet_truthflav.clear();
        deltaR.clear(); jet_tag_btag_DL1r.clear();
        jet_flavindex.clear();

        int njets = jetpf_pt_br[num].size();
        for (int i=0; i<njets; ++i){

            features.push_back((jetpf_pt_br[num][i] - mean_pt)/std_pt);
            features.push_back((jetpf_eta_br[num][i] - mean_eta)/std_eta);
            features.push_back((jetpf_phi_br[num][i] - mean_phi)/std_phi);
            features.push_back((ActualMu_br[num] - mean_mu_actual)/std_mu_actual);
            features.push_back((MET_br[num] - mean_MET)/std_MET);
            // features.push_back(jetpf_truthflav_br[num][i]);
        
	    //  features.push_back((mBB_br[num] - mean_mBB)/std_mBB);


            // Truth Hadron info
            // if (jetpf_truthflav_br[num][i] == 5){
            //     features.push_back((bH_m1_br[num][i] - mean_bH_m)/std_bH_m);
            //     features.push_back((bH_pT1_br[num][i] - mean_bH_pt)/std_bH_pt);
            //     features.push_back((bH_eta1_br[num][i] - mean_bH_eta)/std_bH_eta);
            //     features.push_back((bH_phi1_br[num][i] - mean_bH_phi)/std_bH_phi);
            // } else if (jetpf_truthflav_br[num][i] == 4){
            //     features.push_back((bH_m1_br[num][i] - mean_cH_m)/std_cH_m);
            //     features.push_back((bH_pT1_br[num][i] - mean_cH_pt)/std_cH_pt);
            //     features.push_back((bH_eta1_br[num][i] - mean_cH_eta)/std_cH_eta);
            //     features.push_back((bH_phi1_br[num][i] - mean_cH_phi)/std_cH_phi);
            // } else {
            //     features.push_back(0);
            //     features.push_back(0);
            //     features.push_back(0);
            //     features.push_back(0);
            // }

            // truthflav
            jet_truthflav.push_back(jetpf_truthflav_br[num][i]);


            // deltaR
            for (int j=0; j<njets; ++j){
                if(i==j){ continue; }
                float dR = calculate_deltaR(jetpf_eta_br[num][i], jetpf_phi_br[num][i], jetpf_eta_br[num][j], jetpf_phi_br[num][j]);
                deltaR.push_back((dR - mean_dR)/std_dR);
            }

            //log(pb / (pc*f + (1-f)*pu))
            // jet_tag VR
            double DL1r = TMath::Log(jetpf_DL1r_pb_br[num][i] / (jetpf_DL1r_pc_br[num][i]*f + (1-f)*jetpf_DL1r_pu_br[num][i]));
            if (DL1r > threshold_DL1r){
                jet_tag_btag_DL1r.push_back(1);
            } else {
                jet_tag_btag_DL1r.push_back(0);
            }

            // jet flav index for embedding
            switch (jetpf_truthflav_br[num][i]) {
                case 5:
                    jet_flavindex.push_back(0);
                    break;
                case 4:
                    jet_flavindex.push_back(1);
                    break;
                case 0:
                    jet_flavindex.push_back(2);
                    break;
                case 15:
                    jet_flavindex.push_back(3);
                    break;
//                default:
//                    break;
            }


        } // jet loop ends

        newtree_train->Fill();

    } // event loop ends

    newtree_train->Write();
    outputfile_train->Close();



    std::cout << "boosted event count: " << num_selcted_events << std::endl;
    std::cout << "empty boosted event count: " << num_empty_events << " (these events were removed)" << std::endl;

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


