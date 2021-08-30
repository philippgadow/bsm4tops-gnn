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
        direct, efficiency from custom map, efficiency from NN
        mass and deltaR
     Tagging strategy -
        Direct tag -
            * Exactly two jets are tagged
        Truth tag -
            * all combinations of two jets enter the histogram (allperm)
            * one combination is selected from all the combinations as a representative,
              and it enters the histogram with a weight summing over weights of all the combinations

     Args:
        argv[1]: path to the root file generated with data_prep2pred
        argv[2]: path to the root file containig the flatten tree with NN eficiencies
        argv[3]: which leading
        argv[3]: optional, regions - 'all' | 'one' | 'two'
     */
    
    
    
    // start the timer
    auto t1 = std::chrono::high_resolution_clock::now();

    TFile* data_file;
    TFile* eff_file;

    TString output_filepath;
    TString region = "";
    int which_leading;

    if (argc==4 || argc==5){

        TString eff_filepath = TString(argv[2]);

        data_file = new TFile(TString(argv[1]));
        eff_file  = new TFile(eff_filepath);

        which_leading = atoi(argv[3]);
        
        std::cout << "which leading: " << which_leading << std::endl;

        if (argc == 5){
            TString reg_input = TString(argv[4]);
            if (reg_input == "one"){
                region = "_reg1";
                std::cout << "region chosen: 2-3 jets" << std::endl;
            } else if (reg_input == "two"){
                region = "_reg2";
                std::cout << "region chosen: >=4 jets" << std::endl;
            } else {
                std::cout << "region chosen: None" << std::endl;
            }
        } else {
            std::cout << "region chosen: None" << std::endl;
        }
        
        output_filepath = TString(eff_filepath(0, eff_filepath.Length()-15)) + TString("hist_leading_") + TString(argv[3]) + region + TString(".root");
        
    } else {
        std::cout << "Error: Requires 3 (+ 1 optional) arguments. " << argc-1 << " were given" << std::endl;
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
    TTreeReaderValue<float> EventWeight(data_reader, "EventWeight");
    TTreeReaderValue<std::vector<int>> jet_tag(data_reader, "jet_tag_btag_DL1r");

    TTreeReaderValue<double> mJ(data_reader, "mJ");

    // custom map efficiency
    TTreeReaderValue<std::vector<float>> custom_eff(data_reader, "custom_eff");



    // read the efficiency predictions
    TTreeReader eff_reader("Nominal_flatten", eff_file);
    TTreeReaderValue<double> NN_eff(eff_reader, "efficiency");



    // File to write to
    TFile* outputfile = TFile::Open(output_filepath,"recreate");

    // histograms to fill (direct tag)
    TH1F_custom *mass_hist_direct = new TH1F_custom("mass_hist_direct", "mJ (Direct)", 23, 20, 250);
    TH1F_custom *pt_hist_direct = new TH1F_custom("pt_hist_direct", "Jet pT (Direct)", 23, 20, 250);
    TH1F_custom *dR_hist_direct   = new TH1F_custom("dR_hist_direct", "deltaR (Direct)", 23, 0.4, 4.0);

    // histograms to fill (Efficiency Net)
    TH1F_custom *mass_hist_NN = new TH1F_custom("mass_hist_NN", "mJ", 23, 20, 250);
    TH1F_custom *pt_hist_NN = new TH1F_custom("pt_hist_NN", "Jet pT", 23, 20, 250);
    TH1F_custom *dR_hist_NN   = new TH1F_custom("dR_hist_NN", "deltaR", 23, 0.4, 4.0);

    // histograms to fill (customa efficiency map)
    TH1F_custom *mass_hist_custom = new TH1F_custom("mass_hist_custom", "mJ", 23, 20, 250);
    TH1F_custom *pt_hist_custom = new TH1F_custom("pt_hist_custom", "Jet pT", 23, 20, 250);
    TH1F_custom *dR_hist_custom   = new TH1F_custom("dR_hist_custom", "deltaR", 23, 0.4, 4.0);



    // event loop
    int num = 0;
    while (data_reader.Next()){

        num ++;

        int N = jet_pt->size();
                
        // skip events
        bool skip = false;

        if (N < 2){
            skip = true;
        } else if (N < 4) {
            if (region == "_reg2"){
                skip = true;
            }
        } else {
            if (region == "_reg1"){
                skip = true;
            }
        }


        if ( skip==true ){
            for (int i=0; i<N; i++){
                eff_reader.Next();
            }
            continue;
        }



        // make a vector of NN_eff
        std::vector<double> jet_eff;
        for (int i=0; i<N; i++){
            eff_reader.Next();
            jet_eff.push_back(*NN_eff);
        }



        double mass = *mJ;
//        double dR   = calculate_deltaR(jet_eta->at(0), jet_phi->at(0), jet_eta->at(1), jet_phi->at(1));
        double pT   = jet_pt->at(which_leading)/1000.;

        // Direct tagging (leading jet is tagged)
        if (jet_tag->at(which_leading) == 1){
            mass_hist_direct->Fill(mass, *EventWeight, jet_truthflav->at(which_leading));
//            dR_hist_direct->Fill(dR, *EventWeight, jet_truthflav->at(which_leading));
            pt_hist_direct->Fill(pT, *EventWeight, jet_truthflav->at(which_leading));
        }

        // Truth tagging
        mass_hist_NN->Fill(mass, jet_eff[which_leading] * (*EventWeight), jet_truthflav->at(which_leading));
        pt_hist_NN->Fill(pT, jet_eff[which_leading] * (*EventWeight), jet_truthflav->at(which_leading));
//        dR_hist_NN->Fill(dR, jet_eff[which_leading] * (*EventWeight), jet_truthflav->at(which_leading));

        mass_hist_custom->Fill(mass, custom_eff->at(which_leading) * (*EventWeight), jet_truthflav->at(which_leading));
        pt_hist_custom->Fill(pT, custom_eff->at(which_leading) * (*EventWeight), jet_truthflav->at(which_leading));
//        dR_hist_custom->Fill(dR, custom_eff->at(which_leading) * (*EventWeight), jet_truthflav->at(which_leading));

        if((num+1) % 1000000 == 0){
            std::cout << "Done with " << (num+1)/1000000 << "M events" << std::endl;
        }

    } // end of event loop


    // write the histograms to the root file
    mass_hist_direct->Write();
    mass_hist_NN->Write(); mass_hist_custom->Write();

    pt_hist_direct->Write();
    pt_hist_NN->Write(); pt_hist_custom->Write();

    dR_hist_direct->Write();
    dR_hist_NN->Write(); dR_hist_custom->Write();

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
