#include <iostream>
#include <fstream>
#include <iomanip>
#include <utility>
#include <vector>
#include <algorithm>
#include <chrono>

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



struct TH2F_custom_all {

    int numbins_pt, numbins_eta;
    float max_pt, min_pt, max_eta, min_eta;

    TH2F *bottom, *charm, *light, *tau;

    TH2F_custom_all(int numbins_pt, float min_pt, float max_pt, int numbins_eta, float min_eta, float max_eta){
        bottom = new TH2F("bottom", "bottom (all)" , numbins_pt, min_pt, max_pt, numbins_eta, min_eta, max_eta);
        charm  = new TH2F("charm", "charm (all)" , numbins_pt, min_pt, max_pt, numbins_eta, min_eta, max_eta);
        light  = new TH2F("light", "light (all)" , numbins_pt, min_pt, max_pt, numbins_eta, min_eta, max_eta);
        tau    = new TH2F("tau", "tau (all)" , numbins_pt, min_pt, max_pt, numbins_eta, min_eta, max_eta);
    }

    void Fill(float pt, float eta, float weight, int flav){
        if (flav == 5){
            bottom->Fill(pt, eta, weight);
        } else if (flav == 4){
            charm->Fill(pt, eta, weight);
        } else if (flav == 0){
            light->Fill(pt, eta, weight);
        } else if (flav == 15){
            tau->Fill(pt, eta, weight);
        }
    }

    void Write(){
        bottom->Write(); charm->Write(); light->Write(); tau->Write();
    }
};



struct TH2F_custom_tagged {

    int numbins_pt, numbins_eta;
    float max_pt, min_pt, max_eta, min_eta;

    std::vector<int> tags = {60, 70, 77, 85};
    TH2F *bottoms[4], *charms[4], *lights[4], *taus[4];

    TH2F_custom_tagged(int numbins_pt, float min_pt, float max_pt, int numbins_eta, float min_eta, float max_eta){
        for (unsigned int i=0; i<tags.size(); i++){
            bottoms[i] = new TH2F(Form("bottom_%d", tags[i]), "bottom (tagged, " + TString(tags[i]) + ")",
                                  numbins_pt, min_pt, max_pt, numbins_eta, min_eta, max_eta);
            charms[i]  = new TH2F(Form("charm_%d", tags[i]), "charm (tagged, " + TString(tags[i]) + ")",
                                  numbins_pt, min_pt, max_pt, numbins_eta, min_eta, max_eta);
            lights[i]  = new TH2F(Form("light_%d", tags[i]), "light (tagged, " + TString(tags[i]) + ")",
                                  numbins_pt, min_pt, max_pt, numbins_eta, min_eta, max_eta);
            taus[i]    = new TH2F(Form("tau_%d", tags[i]), "tau (tagged, " + TString(tags[i]) + ")",
                                  numbins_pt, min_pt, max_pt, numbins_eta, min_eta, max_eta);
        }
    }

    void Fill(float pt, float eta, float weight, int flav, int tag){

        auto it = std::find(tags.begin(), tags.end(), tag);
        int index = distance(tags.begin(), it);

        if (flav == 5){
            bottoms[index]->Fill(pt, eta, weight);
        } else if (flav == 4){
            charms[index]->Fill(pt, eta, weight);
        } else if (flav == 0){
            lights[index]->Fill(pt, eta, weight);
        } else if (flav == 15){
            taus[index]->Fill(pt, eta, weight);
        }
    }

    void Write(){
        for (unsigned int i=0; i<tags.size(); i++){
            bottoms[i]->Write(); charms[i]->Write(); lights[i]->Write(); taus[i]->Write();
        }
    }
};





int main(int argc, char *argv[]){

    // start the timer
    auto t1 = std::chrono::high_resolution_clock::now();

    TChain* chain = new TChain("Nominal");
    TString output_filepath;
    
    // args
    if (argc==3){

        TString input_filepath  = TString(argv[1]);
        TString input_filetype = TString(input_filepath(input_filepath.Length()-4, input_filepath.Length()));

        if (input_filetype == TString(".txt")){
            std::ifstream file(input_filepath);
            std::string str;

            std::cout << "Following files will be used -" << std::endl;
            
            while (std::getline(file, str)) {
                std::cout << str << std::endl;
                chain->Add(TString(str));
            }
        } else {
            std::cout << "Following file will be used -" << std::endl;
            std::cout << input_filepath << std::endl;
            chain->Add(input_filepath);
        }
        
        output_filepath = TString(argv[2]);
        
    } else {
        std::cout << "Error: Requires exactly 2 arguments. " << argc-1 << " were given" << std::endl;
        std::cout << "Exiting..." << std::endl;
        return 0;
    }

    std::cout << "2d map will be saved at " << output_filepath << std::endl;

    
    
    Reader* reader = new Reader(chain);

    // Turn off all the branches except the ones needed
    chain->SetBranchStatus("*",0);

    chain->SetBranchStatus("jetpf_pt", 1 );
    chain->SetBranchStatus("jetpf_eta", 1 );
    chain->SetBranchStatus("jetpf_truthflav", 1 );
    
    // DL1r info
    chain->SetBranchStatus("jetpf_DL1r_pb", 1 );
    chain->SetBranchStatus("jetpf_DL1r_pc", 1 );
    chain->SetBranchStatus("jetpf_DL1r_pu", 1 );

    // for event selection
    chain->SetBranchStatus("mJ", 1 );

    
    // Create the output file, and the histogrmas
    TFile* outputfile = TFile::Open(output_filepath,"recreate");

    TH2F_custom_all *hist_all = new TH2F_custom_all(38, 10e3, 400e3, 13, 0, 2.5);
    TH2F_custom_tagged *hist_tagged = new TH2F_custom_tagged(38, 10e3, 400e3, 13, 0, 2.5);

    // working points
    float threshold_60 = 4, f_60 = 0.018;
    float threshold_70 = 3.515, f_70 = 0.018;
    float threshold_77 = 4, f_77 = 0.018;
    float threshold_85 = 4, f_85 = 0.018;



    // working with less number of events
    int num_entries = chain->GetEntries();
    std::cout << "total entries: " << num_entries << std::endl;

    int num_selcted_events = 0, num_empty_events=0;
    
    // event loop
    std::cout << "Looping over the events..." << std::endl;
    for (int num = 0; num < num_entries; ++num){
        
        chain->GetEntry(num);
        reader->GetEntry(num);
        
        if((num+1) % int(1e6) == 0){
            std::cout << "Done with " << (num+1)/1e6 << "M events" <<std::endl;
        }
        
        // skip if not a boosted event
        if (reader->mJ <= 0){
            continue;
        }
        
        // skip empty events
        int njets = reader->jetpf_pt->size();
        if (njets == 0){
            continue;
        }
        
        // skip one jet events
        if (njets == 1){
            continue;
        }
        
        // skip high pT (extreme) event
        if (reader->jetpf_pt->at(0) > 1E6){
            continue;
        }

        num_selcted_events++;
        
        float event_weight = 1.0; //reader->weight_mc * reader->weight_pileup;
        for (int i=0; i<njets; ++i){

            float jet_pt = reader->jetpf_pt->at(i);
            float jet_eta = TMath::Abs(reader->jetpf_eta->at(i));
            int jet_truthflav = reader->jetpf_truthflav->at(i);

            // fill the histogrms with all jets
            hist_all->Fill(jet_pt, jet_eta, event_weight, jet_truthflav);

            // fill the histograms with tagged jets
            float DL1r_60 = TMath::Log(reader->jetpf_DL1r_pb->at(i) / (reader->jetpf_DL1r_pc->at(i)*f_60 + (1-f_60)*reader->jetpf_DL1r_pu->at(i)));
            if (DL1r_60 > threshold_60){
                hist_tagged->Fill(jet_pt, jet_eta, event_weight, jet_truthflav, 60);
            }

            float DL1r_70 = TMath::Log(reader->jetpf_DL1r_pb->at(i) / (reader->jetpf_DL1r_pc->at(i)*f_70 + (1-f_70)*reader->jetpf_DL1r_pu->at(i)));
            if (DL1r_70 > threshold_70){
                hist_tagged->Fill(jet_pt, jet_eta, event_weight, jet_truthflav, 70);
            }

            float DL1r_77 = TMath::Log(reader->jetpf_DL1r_pb->at(i) / (reader->jetpf_DL1r_pc->at(i)*f_77 + (1-f_77)*reader->jetpf_DL1r_pu->at(i)));
            if (DL1r_77 > threshold_77){
                hist_tagged->Fill(jet_pt, jet_eta, event_weight, jet_truthflav, 77);
            }

            float DL1r_85 = TMath::Log(reader->jetpf_DL1r_pb->at(i) / (reader->jetpf_DL1r_pc->at(i)*f_85 + (1-f_85)*reader->jetpf_DL1r_pu->at(i)));
            if (DL1r_85 > threshold_85){
                hist_tagged->Fill(jet_pt, jet_eta, event_weight, jet_truthflav, 85);
            }
        }
                
    } // end of event loop
    
    
    
    // calculate efficiencies
    TString wps[4]   = {"60", "70", "77", "85"};
    
    TH2F* total_hist  = hist_all->bottom;
    for (int wp_num=0; wp_num<4; wp_num++){
        TH2F* passed_hist = hist_tagged->bottoms[wp_num];
        TH2F* eff = (TH2F*)total_hist->Clone(TString("bottom") + "_" + wps[wp_num] + "eff");
        eff->Divide(passed_hist, total_hist, 1, 1, "B");
        eff->Write();
    }
    
    delete total_hist;
    total_hist  = hist_all->charm;
    for (int wp_num=0; wp_num<4; wp_num++){
        TH2F* passed_hist = hist_tagged->charms[wp_num];
        TH2F* eff = (TH2F*)total_hist->Clone(TString("charm") + "_" + wps[wp_num] + "eff");
        eff->Divide(passed_hist, total_hist, 1, 1, "B");
        eff->Write();
    }

    delete total_hist;
    total_hist  = hist_all->light;
    for (int wp_num=0; wp_num<4; wp_num++){
        TH2F* passed_hist = hist_tagged->lights[wp_num];
        TH2F* eff = (TH2F*)total_hist->Clone(TString("light") + "_" + wps[wp_num] + "eff");
        eff->Divide(passed_hist, total_hist, 1, 1, "B");
        eff->Write();
    }

    delete total_hist;
    total_hist  = hist_all->tau;
    for (int wp_num=0; wp_num<4; wp_num++){
        TH2F* passed_hist = hist_tagged->taus[wp_num];
        TH2F* eff = (TH2F*)total_hist->Clone(TString("tau") + "_" + wps[wp_num] + "eff");
        eff->Divide(passed_hist, total_hist, 1, 1, "B");
        eff->Write();
    }

    
    outputfile->Close();
    
    std::cout << "selected event count: " << num_selcted_events << std::endl;
    
    // computation time
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();

    int total_min = (int)(duration / 60000000);
    float seconds = ((float)(duration / 1000000.)) - (float(total_min) * 60.);
    int min       = total_min % 60;
    int hours     = total_min / 60;

    std::cout << std::endl;
    std::cout << "Time taken: " << hours << " hrs " << min << " min "
              << seconds << " seconds " << std::endl;

    return 0;
}

