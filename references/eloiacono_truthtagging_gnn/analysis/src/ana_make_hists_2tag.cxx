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


// weight issue
struct TH1F_custom {

    int numbins;
    double max, min;
    TString name, title;

    TH1F *hist_bb, *hist_cc, *hist_ll, *hist_bc, *hist_bl, *hist_cl;
    TH1F_custom(TString name, TString title, int numbins, double min, double max){
        hist_bb = new TH1F(name + "_bb", title + " (bottom bottom)" , numbins, min, max);
        hist_cc = new TH1F(name + "_cc", title + " (charm charm)" , numbins, min, max);
        hist_ll = new TH1F(name + "_ll", title + " (light light)" , numbins, min, max);
        hist_bc = new TH1F(name + "_bc", title + " (bottom charm)" , numbins, min, max);
        hist_bl = new TH1F(name + "_bl", title + " (bottom light)" , numbins, min, max);
        hist_cl = new TH1F(name + "_cl", title + " (charm light)" , numbins, min, max);
    }

    void Fill(double val, double weight, int flav1, int flav2){
        if (flav1 == 5 && flav2 == 5){
            hist_bb->Fill(val, weight);
        } else if (flav1 == 4 && flav2 == 4){
            hist_cc->Fill(val, weight);
        } else if (flav1 == 0 && flav2 == 0){
            hist_ll->Fill(val, weight);
        } else if ((flav1 == 5 && flav2 == 4) || (flav1 == 4 && flav2 == 5)){
            hist_bc->Fill(val, weight);
        } else if ((flav1 == 5 && flav2 == 0) || (flav1 == 0 && flav2 == 5)){
            hist_bl->Fill(val, weight);
        } else if ((flav1 == 4 && flav2 == 0) || (flav1 == 0 && flav2 == 4)){
            hist_cl->Fill(val, weight);
        }
    }

    void Write(){
        hist_bb->Write(); hist_cc->Write(); hist_ll->Write();
        hist_bc->Write(); hist_bl->Write(); hist_cl->Write();
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
        argv[3]: optional, regions - 'all' | 'one' | 'two'
     */
    
    
    
    // start the timer
    auto t1 = std::chrono::high_resolution_clock::now();

    TFile* data_file;
    TFile* eff_file;

    TString output_filepath;
    TString region = "";

    if (argc==3 || argc==4){

        TString eff_filepath = TString(argv[2]);

        data_file = new TFile(TString(argv[1]));
        eff_file  = new TFile(eff_filepath);
                
        if (argc == 4){
            TString reg_input = TString(argv[3]);
            if (reg_input == "one"){
                region = "_reg1";
                std::cout << "region chosen: 2 jets" << std::endl;
            } else if (reg_input == "two"){
                region = "_reg2";
                std::cout << "region chosen: 3 jets" << std::endl;
            } else {
                std::cout << "region chosen: None" << std::endl;
            }
        
        } else {
            std::cout << "region chosen: None" << std::endl;
        }
        
        
        output_filepath = TString(eff_filepath(0, eff_filepath.Length()-15)) + TString("hist") + region + TString(".root");
        
    } else {
        std::cout << "Error: Requires 2 (+ 1 optional) arguments. " << argc-1 << " were given" << std::endl;
        std::cout << "Exiting..." << std::endl;
        return 0;
    }

    std::cout << "histograms will be saved at " << output_filepath << std::endl;



    // read the data file
    TTreeReader data_reader("Nominal", data_file);

    TTreeReaderValue<std::vector<float>> jet_pt(data_reader, "jet_pt");
    TTreeReaderValue<std::vector<float>> jet_eta(data_reader, "jet_eta");
    TTreeReaderValue<std::vector<float>> jet_phi(data_reader, "jet_phi");
    TTreeReaderValue<double> mJ(data_reader, "mJ");

    TTreeReaderValue<std::vector<int>> jet_truthflav(data_reader, "jet_truthflav");
    TTreeReaderValue<float> EventWeight(data_reader, "EventWeight");
    TTreeReaderValue<std::vector<int>> jet_tag(data_reader, "jet_tag_btag_DL1r");

    // custom map efficiency
    TTreeReaderValue<std::vector<float>> custom_eff(data_reader, "custom_eff");

    TTreeReaderValue<float> MET(data_reader, "MET");


    // read the efficiency predictions
    TTreeReader eff_reader("Nominal_flatten", eff_file);
    TTreeReaderValue<double> NN_eff(eff_reader, "efficiency");


    
    // File to write to
    TFile* outputfile = TFile::Open(output_filepath,"recreate");

    // histograms to fill (direct tag)
    TH1F_custom *mass_hist_direct = new TH1F_custom("mass_hist_direct", "Event Mass (Direct)", 23, 20, 250);
    TH1F_custom *dR_hist_direct = new TH1F_custom("dR_hist_direct", "deltaR (Direct)", 23, 0, 1.2);

    // histograms to fill (Efficiency Net)
    TH1F_custom *mass_hist_NN = new TH1F_custom("mass_hist_NN", "Event Mass", 23, 20, 250);
    TH1F_custom *mass_hist_allperm_NN = new TH1F_custom("mass_hist_allperm_NN", "Event Mass (All Permutations)", 23, 20, 250);

    TH1F_custom *dR_hist_NN = new TH1F_custom("dR_hist_NN", "deltaR", 23, 0, 1.2);
    TH1F_custom *dR_hist_allperm_NN = new TH1F_custom("dR_hist_allperm_NN", "deltaR (All Permutations)", 23, 0, 1.2);

    // histograms to fill (customa efficiency map)
    TH1F_custom *mass_hist_custom = new TH1F_custom("mass_hist_custom", "Event Mass", 23, 20, 250);
    TH1F_custom *mass_hist_allperm_custom = new TH1F_custom("mass_hist_allperm_custom", "Event Mass (All Permutations)", 23, 20, 250);

    TH1F_custom *dR_hist_custom = new TH1F_custom("dR_hist_custom", "deltaR", 23, 0, 1.2);
    TH1F_custom *dR_hist_allperm_custom = new TH1F_custom("dR_hist_allperm_custom", "deltaR (All Permutations)", 23, 0, 1.2);



    double eventMass, eventdR;

    std::default_random_engine generator;

    // event loop
    int num = 0;
    while (data_reader.Next()){

        num ++;
        int N = jet_pt->size();
        
        
        //*****************//
        // Event Selection //
        //*****************//

        // skip events
        bool skip = false;

        if (N < 2){
            skip = true;
        } else if (N == 2) {
            if (region == "_reg2"){
                skip = true;
            }
        } else if (N == 3) {
            if (region == "_reg1"){
                skip = true;
            }
        }
         
        if (skip == false){
            if (*MET < 250){
                skip= true;
            }
        }
        
        
        if ( skip==true ){
            for (int i=0; i<N; i++){
                eff_reader.Next();
            }
            continue;
        }
        
    
        
        //************************************************************************//
        // make the TLorentzVectors (setting jet_e=0, casue we only calculate dR) //
        //************************************************************************//

        std::vector<TLorentzVector> jets;
        std::vector<int> jet_istag;

        std::vector<double> jet_eff;

        // loop over the jets
        
        for (int i=0; i<N; i++){
            TLorentzVector v;
            v.SetPtEtaPhiE (jet_pt->at(i)/1000.0, jet_eta->at(i), jet_phi->at(i), 0);
            jets.push_back(v);

            eff_reader.Next();
            jet_eff.push_back(*NN_eff);
        }


        
        //****************//
        // Direct Tagging //
        //****************//

        // Direct tagging (exactly two tagged)
        int num_tagged = std::accumulate(jet_tag->begin(), jet_tag->end(), 0);
        if (num_tagged ==  2){
            // find the two tagged jets
            std::vector<int> tag_num;
            for (int i=0; i<N; i++){
                if (jet_tag->at(i) == 1){
                    tag_num.push_back(i);
                }
            }
            double deltaR = (jets[tag_num[0]]).DeltaR(jets[tag_num[1]]);

            mass_hist_direct->Fill(*mJ, *EventWeight, jet_truthflav->at(tag_num[0]), jet_truthflav->at(tag_num[1]));
            dR_hist_direct->Fill(deltaR, *EventWeight, jet_truthflav->at(tag_num[0]), jet_truthflav->at(tag_num[1]));
        }



        //*************//
        // Permutation //
        //*************//
        
        std::vector<double> dR_perm;
        std::vector<double> weight_perm_NN, weight_perm_custom;
        std::vector<std::vector<int>> flav_perm;

        // for all the combinations
        for (int i=0; i<N; i++){
            for (int j=i+1; j<N; j++){

                double dR = jets[i].DeltaR(jets[j]);
                dR_perm.push_back(dR);

                std::vector<int> tmp;
                tmp.push_back(jet_truthflav->at(i)); tmp.push_back(jet_truthflav->at(j));
                flav_perm.push_back(tmp);

                double per_eff_NN = jet_eff[i] * jet_eff[j];
                double per_eff_custom = custom_eff->at(i) * custom_eff->at(j);
                for (int k=0; k<N; k++){
                    if (k==i || k==j){
                        continue;
                    }
                    per_eff_NN *= (1 - jet_eff[k]);
                    per_eff_custom *= (1 - custom_eff->at(k));
                }
                weight_perm_NN.push_back(per_eff_NN);
                weight_perm_custom.push_back(per_eff_custom);

                // fill the allperm histograms
                mass_hist_allperm_NN->Fill(*mJ, per_eff_NN * (*EventWeight), jet_truthflav->at(i), jet_truthflav->at(j));
                dR_hist_allperm_NN->Fill(dR, per_eff_NN * (*EventWeight), jet_truthflav->at(i), jet_truthflav->at(j));

                mass_hist_allperm_custom->Fill(*mJ, per_eff_custom * (*EventWeight), jet_truthflav->at(i), jet_truthflav->at(j));
                dR_hist_allperm_custom->Fill(dR, per_eff_custom * (*EventWeight), jet_truthflav->at(i), jet_truthflav->at(j));
            }
        }


        
        //************************//
        // Choose one permutation //
        //************************//
        
        // choose one combination (Efficiency Net)
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        generator.seed(seed);
        std::discrete_distribution<int> distribution (weight_perm_NN.begin(), weight_perm_NN.end());

        int perm_num = distribution(generator);

        eventMass = *mJ;
        eventdR   = dR_perm[perm_num];
        std::vector<int> flavs = flav_perm[perm_num];

        // fill the one comb histograms (Efficiency Net)
        double weight_NN = std::accumulate(weight_perm_NN.begin(), weight_perm_NN.end(), 0.0);

        mass_hist_NN->Fill(eventMass, weight_NN * (*EventWeight), flavs[0], flavs[1]);
        dR_hist_NN->Fill(eventdR, weight_NN * (*EventWeight), flavs[0], flavs[1]);


        // choose one combination (custom map)
        seed = std::chrono::system_clock::now().time_since_epoch().count();
        generator.seed(seed);
        std::discrete_distribution<int> distribution2 (weight_perm_custom.begin(), weight_perm_custom.end());

        perm_num = distribution2(generator);

        eventMass = *mJ;
        eventdR   = dR_perm[perm_num];
        flavs = flav_perm[perm_num];

        // fill the one comb histograms (custom map)
        double weight_custom = std::accumulate(weight_perm_custom.begin(), weight_perm_custom.end(), 0.0);

        mass_hist_custom->Fill(eventMass, weight_custom * (*EventWeight), flavs[0], flavs[1]);
        dR_hist_custom->Fill(eventdR, weight_custom * (*EventWeight), flavs[0], flavs[1]);

        if((num+1) % 1000000 == 0){
            std::cout << "Done with " << (num+1)/1000000 << "M events" << std::endl;
        }

    } // end of event loop

    
    // write the histograms to the root file
    mass_hist_direct->Write();
    mass_hist_NN->Write(); mass_hist_allperm_NN->Write();
    mass_hist_custom->Write(); mass_hist_allperm_custom->Write();

    dR_hist_direct->Write();
    dR_hist_NN->Write(); dR_hist_allperm_NN->Write();
    dR_hist_custom->Write(); dR_hist_allperm_custom->Write();

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
