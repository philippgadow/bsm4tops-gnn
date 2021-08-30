//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Mon Dec  7 12:40:42 2020 by ROOT version 6.18/04
// from TTree Nominal/Nominal
// found on file: Znunu.root
//////////////////////////////////////////////////////////

#ifndef Reader_h
#define Reader_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>

// Header file for the classes stored in the TTree if any.
#include "string"
#include "vector"
#include "vector"
#include "vector"

using namespace std;

class Reader {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
   Int_t           fCurrent; //!current Tree number in a TChain

// Fixed size dimensions of array or collections stored in the TTree if any.

   // Declaration of leaf types
   Int_t           nJets;
   Int_t           nTags;
   Int_t           nTaus;
   Int_t           flavB1;
   Int_t           flavB2;
   Int_t           flavJ3;
   Int_t           nbJets;
   Int_t           isMerged;
   Int_t           nFwdJets;
   Int_t           nSigJets;
   Int_t           nTrkJets;
   Int_t           RunNumber;
   Int_t           isOverlap;
   Int_t           nBTrkJets;
   Int_t           isResolved;
   Int_t           nbTagsInFJ;
   Int_t           passVRJetOR;
   Int_t           nTrkjetsInFJ;
   Int_t           MCChannelNumber;
   Int_t           nbTagsOutsideFJ;
   Int_t           Njets_truth_pTjet30;
   Int_t           IsTaggedTrkJet1InLeadFJ;
   Int_t           IsTaggedTrkJet2InLeadFJ;
   Int_t           IsTaggedTrkJet3InLeadFJ;
   Int_t           IsTaggedTrkJet1NotInLeadFJ;
   UInt_t          nFatJets;
   UInt_t          NAdditionalCaloJets;
   UInt_t          NMatchedTrackJetLeadFatJet;
   UInt_t          NBTagMatchedTrackJetLeadFatJet;
   UInt_t          NBTagUnmatchedTrackJetLeadFatJet;
   ULong64_t       EventNumber;
   Float_t         MET;
   Float_t         mB1;
   Float_t         mB2;
   Float_t         mBB;
   Float_t         mJ3;
   Float_t         pTV;
   Float_t         dRBB;
   Float_t         mBBJ;
   Float_t         pTB1;
   Float_t         pTB2;
   Float_t         pTBB;
   Float_t         pTJ3;
   Float_t         etaB1;
   Float_t         etaB2;
   Float_t         etaBB;
   Float_t         etaJ3;
   Float_t         pTBBJ;
   Float_t         phiB1;
   Float_t         phiB2;
   Float_t         phiBB;
   Float_t         phiJ3;
   Float_t         BTagSF;
   Float_t         dEtaBB;
   Float_t         dPhiBB;
   Float_t         metSig;
   Float_t         softMET;
   Float_t         ActualMu;
   Float_t         MV2c10B1;
   Float_t         MV2c10B2;
   Float_t         PUWeight;
   Float_t         mH_truth;
   Float_t         mV_truth;
   Float_t         AverageMu;
   Float_t         HTBoosted;
   Float_t         JVTWeight;
   Float_t         TriggerSF;
   Float_t         pTH_truth;
   Float_t         pTV_truth;
   Float_t         sumPtJets;
   Float_t         LumiWeight;
   Float_t         etaH_truth;
   Float_t         etaV_truth;
   Float_t         phiH_truth;
   Float_t         phiV_truth;
   Float_t         EventWeight;
   Float_t         bin_MV2c10B1;
   Float_t         bin_MV2c10B2;
   Float_t         bin_MV2c10J3;
   Float_t         MCEventWeight;
   Float_t         pTAddCaloJets;
   Float_t         ActualMuScaled;
   Float_t         AverageMuScaled;
   Double_t        C2;
   Double_t        D2;
   Double_t        HT;
   Double_t        mJ;
   Double_t        mVJ;
   Double_t        mva;
   Double_t        pTJ;
   Double_t        EtaJ;
   Double_t        PhiJ;
   Double_t        Tau21;
   Double_t        phiMET;
   Double_t        phiMPT;
   Double_t        SUSYMET;
   Double_t        deltaYVJ;
   Double_t        MET_Track;
   Double_t        dPhiMETMPT;
   Double_t        mvadiboson;
   Double_t        MindPhiMETJet;
   Double_t        absdeltaPhiVJ;
   Double_t        dEtabTrkJbTrkJ;
   Double_t        MTrkJet1InLeadFJ;
   Double_t        MTrkJet2InLeadFJ;
   Double_t        MTrkJet3InLeadFJ;
   Double_t        bin_MV2c10BTrkJ1;
   Double_t        bin_MV2c10BTrkJ2;
   Double_t        deltaRbTrkJbTrkJ;
   Double_t        PtTrkJet1InLeadFJ;
   Double_t        PtTrkJet2InLeadFJ;
   Double_t        PtTrkJet3InLeadFJ;
   Double_t        EtaTrkJet1InLeadFJ;
   Double_t        EtaTrkJet2InLeadFJ;
   Double_t        EtaTrkJet3InLeadFJ;
   Double_t        MV2TrkJet1InLeadFJ;
   Double_t        MV2TrkJet2InLeadFJ;
   Double_t        PhiTrkJet1InLeadFJ;
   Double_t        PhiTrkJet2InLeadFJ;
   Double_t        PhiTrkJet3InLeadFJ;
   Double_t        deltaPhibTrkJbTrkJ;
   Double_t        PtTrkJet1NotInLeadFJ;
   Double_t        dPhiMETdijetResolved;
   Double_t        DL1_pbTrkJet1InLeadFJ;
   Double_t        DL1_pbTrkJet2InLeadFJ;
   Double_t        DL1_pcTrkJet1InLeadFJ;
   Double_t        DL1_pcTrkJet2InLeadFJ;
   Double_t        DL1_puTrkJet1InLeadFJ;
   Double_t        DL1_puTrkJet2InLeadFJ;
   Double_t        EtaTrkJet1NotInLeadFJ;
   Double_t        PhiTrkJet1NotInLeadFJ;
   string          *sample;
   string          *Description;
   string          *EventFlavor;
   vector<int>     *bH_pdgid1;
   vector<int>     *jetvr_nBaGA;
   vector<int>     *jetpf_truthflav;
   vector<int>     *jetvr_truthflav;
   vector<int>     *index_Pflow_to_VR;
   vector<float>   *bH_m1;
   vector<float>   *bH_pT1;
   vector<float>   *bH_eta1;
   vector<float>   *bH_phi1;
   vector<float>   *jetpf_pt;
   vector<float>   *jetvr_pt;
   vector<float>   *jetpf_eta;
   vector<float>   *jetpf_phi;
   vector<float>   *jetvr_eta;
   vector<float>   *jetvr_phi;
   vector<float>   *jetpf_mv2c10;
   vector<float>   *jetvr_mv2c10;
   vector<double>  *jetpf_DL1r_pb;
   vector<double>  *jetpf_DL1r_pc;
   vector<double>  *jetpf_DL1r_pu;
   vector<double>  *jetvr_DL1r_pb;
   vector<double>  *jetvr_DL1r_pc;
   vector<double>  *jetvr_DL1r_pu;

   // List of branches
   TBranch        *b_nJets;   //!
   TBranch        *b_nTags;   //!
   TBranch        *b_nTaus;   //!
   TBranch        *b_flavB1;   //!
   TBranch        *b_flavB2;   //!
   TBranch        *b_flavJ3;   //!
   TBranch        *b_nbJets;   //!
   TBranch        *b_isMerged;   //!
   TBranch        *b_nFwdJets;   //!
   TBranch        *b_nSigJets;   //!
   TBranch        *b_nTrkJets;   //!
   TBranch        *b_RunNumber;   //!
   TBranch        *b_isOverlap;   //!
   TBranch        *b_nBTrkJets;   //!
   TBranch        *b_isResolved;   //!
   TBranch        *b_nbTagsInFJ;   //!
   TBranch        *b_passVRJetOR;   //!
   TBranch        *b_nTrkjetsInFJ;   //!
   TBranch        *b_MCChannelNumber;   //!
   TBranch        *b_nbTagsOutsideFJ;   //!
   TBranch        *b_Njets_truth_pTjet30;   //!
   TBranch        *b_IsTaggedTrkJet1InLeadFJ;   //!
   TBranch        *b_IsTaggedTrkJet2InLeadFJ;   //!
   TBranch        *b_IsTaggedTrkJet3InLeadFJ;   //!
   TBranch        *b_IsTaggedTrkJet1NotInLeadFJ;   //!
   TBranch        *b_nFatJets;   //!
   TBranch        *b_NAdditionalCaloJets;   //!
   TBranch        *b_NMatchedTrackJetLeadFatJet;   //!
   TBranch        *b_NBTagMatchedTrackJetLeadFatJet;   //!
   TBranch        *b_NBTagUnmatchedTrackJetLeadFatJet;   //!
   TBranch        *b_EventNumber;   //!
   TBranch        *b_MET;   //!
   TBranch        *b_mB1;   //!
   TBranch        *b_mB2;   //!
   TBranch        *b_mBB;   //!
   TBranch        *b_mJ3;   //!
   TBranch        *b_pTV;   //!
   TBranch        *b_dRBB;   //!
   TBranch        *b_mBBJ;   //!
   TBranch        *b_pTB1;   //!
   TBranch        *b_pTB2;   //!
   TBranch        *b_pTBB;   //!
   TBranch        *b_pTJ3;   //!
   TBranch        *b_etaB1;   //!
   TBranch        *b_etaB2;   //!
   TBranch        *b_etaBB;   //!
   TBranch        *b_etaJ3;   //!
   TBranch        *b_pTBBJ;   //!
   TBranch        *b_phiB1;   //!
   TBranch        *b_phiB2;   //!
   TBranch        *b_phiBB;   //!
   TBranch        *b_phiJ3;   //!
   TBranch        *b_BTagSF;   //!
   TBranch        *b_dEtaBB;   //!
   TBranch        *b_dPhiBB;   //!
   TBranch        *b_metSig;   //!
   TBranch        *b_softMET;   //!
   TBranch        *b_ActualMu;   //!
   TBranch        *b_MV2c10B1;   //!
   TBranch        *b_MV2c10B2;   //!
   TBranch        *b_PUWeight;   //!
   TBranch        *b_mH_truth;   //!
   TBranch        *b_mV_truth;   //!
   TBranch        *b_AverageMu;   //!
   TBranch        *b_HTBoosted;   //!
   TBranch        *b_JVTWeight;   //!
   TBranch        *b_TriggerSF;   //!
   TBranch        *b_pTH_truth;   //!
   TBranch        *b_pTV_truth;   //!
   TBranch        *b_sumPtJets;   //!
   TBranch        *b_LumiWeight;   //!
   TBranch        *b_etaH_truth;   //!
   TBranch        *b_etaV_truth;   //!
   TBranch        *b_phiH_truth;   //!
   TBranch        *b_phiV_truth;   //!
   TBranch        *b_EventWeight;   //!
   TBranch        *b_bin_MV2c10B1;   //!
   TBranch        *b_bin_MV2c10B2;   //!
   TBranch        *b_bin_MV2c10J3;   //!
   TBranch        *b_MCEventWeight;   //!
   TBranch        *b_pTAddCaloJets;   //!
   TBranch        *b_ActualMuScaled;   //!
   TBranch        *b_AverageMuScaled;   //!
   TBranch        *b_C2;   //!
   TBranch        *b_D2;   //!
   TBranch        *b_HT;   //!
   TBranch        *b_mJ;   //!
   TBranch        *b_mVJ;   //!
   TBranch        *b_mva;   //!
   TBranch        *b_pTJ;   //!
   TBranch        *b_EtaJ;   //!
   TBranch        *b_PhiJ;   //!
   TBranch        *b_Tau21;   //!
   TBranch        *b_phiMET;   //!
   TBranch        *b_phiMPT;   //!
   TBranch        *b_SUSYMET;   //!
   TBranch        *b_deltaYVJ;   //!
   TBranch        *b_MET_Track;   //!
   TBranch        *b_dPhiMETMPT;   //!
   TBranch        *b_mvadiboson;   //!
   TBranch        *b_MindPhiMETJet;   //!
   TBranch        *b_absdeltaPhiVJ;   //!
   TBranch        *b_dEtabTrkJbTrkJ;   //!
   TBranch        *b_MTrkJet1InLeadFJ;   //!
   TBranch        *b_MTrkJet2InLeadFJ;   //!
   TBranch        *b_MTrkJet3InLeadFJ;   //!
   TBranch        *b_bin_MV2c10BTrkJ1;   //!
   TBranch        *b_bin_MV2c10BTrkJ2;   //!
   TBranch        *b_deltaRbTrkJbTrkJ;   //!
   TBranch        *b_PtTrkJet1InLeadFJ;   //!
   TBranch        *b_PtTrkJet2InLeadFJ;   //!
   TBranch        *b_PtTrkJet3InLeadFJ;   //!
   TBranch        *b_EtaTrkJet1InLeadFJ;   //!
   TBranch        *b_EtaTrkJet2InLeadFJ;   //!
   TBranch        *b_EtaTrkJet3InLeadFJ;   //!
   TBranch        *b_MV2TrkJet1InLeadFJ;   //!
   TBranch        *b_MV2TrkJet2InLeadFJ;   //!
   TBranch        *b_PhiTrkJet1InLeadFJ;   //!
   TBranch        *b_PhiTrkJet2InLeadFJ;   //!
   TBranch        *b_PhiTrkJet3InLeadFJ;   //!
   TBranch        *b_deltaPhibTrkJbTrkJ;   //!
   TBranch        *b_PtTrkJet1NotInLeadFJ;   //!
   TBranch        *b_dPhiMETdijetResolved;   //!
   TBranch        *b_DL1_pbTrkJet1InLeadFJ;   //!
   TBranch        *b_DL1_pbTrkJet2InLeadFJ;   //!
   TBranch        *b_DL1_pcTrkJet1InLeadFJ;   //!
   TBranch        *b_DL1_pcTrkJet2InLeadFJ;   //!
   TBranch        *b_DL1_puTrkJet1InLeadFJ;   //!
   TBranch        *b_DL1_puTrkJet2InLeadFJ;   //!
   TBranch        *b_EtaTrkJet1NotInLeadFJ;   //!
   TBranch        *b_PhiTrkJet1NotInLeadFJ;   //!
   TBranch        *b_sample;   //!
   TBranch        *b_Description;   //!
   TBranch        *b_EventFlavor;   //!
   TBranch        *b_bH_pdgid1;   //!
   TBranch        *b_jetvr_nBaGA;   //!
   TBranch        *b_jetpf_truthflav;   //!
   TBranch        *b_jetvr_truthflav;   //!
   TBranch        *b_index_Pflow_to_VR;   //!
   TBranch        *b_bH_m1;   //!
   TBranch        *b_bH_pT1;   //!
   TBranch        *b_bH_eta1;   //!
   TBranch        *b_bH_phi1;   //!
   TBranch        *b_jetpf_pt;   //!
   TBranch        *b_jetvr_pt;   //!
   TBranch        *b_jetpf_eta;   //!
   TBranch        *b_jetpf_phi;   //!
   TBranch        *b_jetvr_eta;   //!
   TBranch        *b_jetvr_phi;   //!
   TBranch        *b_jetpf_mv2c10;   //!
   TBranch        *b_jetvr_mv2c10;   //!
   TBranch        *b_jetpf_DL1r_pb;   //!
   TBranch        *b_jetpf_DL1r_pc;   //!
   TBranch        *b_jetpf_DL1r_pu;   //!
   TBranch        *b_jetvr_DL1r_pb;   //!
   TBranch        *b_jetvr_DL1r_pc;   //!
   TBranch        *b_jetvr_DL1r_pu;   //!

   Reader(TTree *tree=0);
   virtual ~Reader();
   virtual Int_t    Cut(Long64_t entry);
   virtual Int_t    GetEntry(Long64_t entry);
   virtual Long64_t LoadTree(Long64_t entry);
   virtual void     Init(TTree *tree);
   virtual void     Loop();
   virtual Bool_t   Notify();
   virtual void     Show(Long64_t entry = -1);
};

#endif

#ifdef Reader_cxx
Reader::Reader(TTree *tree) : fChain(0) 
{
// if parameter tree is not specified (or zero), connect the file
// used to generate this class and read the Tree.
   if (tree == 0) {
      TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("/home/loiacoel/GNN/ttbar_nonallhad.root");
      if (!f || !f->IsOpen()) {
         f = new TFile("ttbar_nonallhad.root");
      }
      f->GetObject("Nominal",tree);

   }
   Init(tree);
}

Reader::~Reader()
{
   if (!fChain) return;
   delete fChain->GetCurrentFile();
}

Int_t Reader::GetEntry(Long64_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}
Long64_t Reader::LoadTree(Long64_t entry)
{
// Set the environment to read one entry
   if (!fChain) return -5;
   Long64_t centry = fChain->LoadTree(entry);
   if (centry < 0) return centry;
   if (fChain->GetTreeNumber() != fCurrent) {
      fCurrent = fChain->GetTreeNumber();
      Notify();
   }
   return centry;
}

void Reader::Init(TTree *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the branch addresses and branch
   // pointers of the tree will be set.
   // It is normally not necessary to make changes to the generated
   // code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running on PROOF
   // (once per file to be processed).

   // Set object pointer
   sample = 0;
   Description = 0;
   EventFlavor = 0;
   bH_pdgid1 = 0;
   jetvr_nBaGA = 0;
   jetpf_truthflav = 0;
   jetvr_truthflav = 0;
   index_Pflow_to_VR = 0;
   bH_m1 = 0;
   bH_pT1 = 0;
   bH_eta1 = 0;
   bH_phi1 = 0;
   jetpf_pt = 0;
   jetvr_pt = 0;
   jetpf_eta = 0;
   jetpf_phi = 0;
   jetvr_eta = 0;
   jetvr_phi = 0;
   jetpf_mv2c10 = 0;
   jetvr_mv2c10 = 0;
   jetpf_DL1r_pb = 0;
   jetpf_DL1r_pc = 0;
   jetpf_DL1r_pu = 0;
   jetvr_DL1r_pb = 0;
   jetvr_DL1r_pc = 0;
   jetvr_DL1r_pu = 0;
   // Set branch addresses and branch pointers
   if (!tree) return;
   fChain = tree;
   fCurrent = -1;
   fChain->SetMakeClass(1);

   fChain->SetBranchAddress("nJets", &nJets, &b_nJets);
   fChain->SetBranchAddress("nTags", &nTags, &b_nTags);
   fChain->SetBranchAddress("nTaus", &nTaus, &b_nTaus);
   fChain->SetBranchAddress("flavB1", &flavB1, &b_flavB1);
   fChain->SetBranchAddress("flavB2", &flavB2, &b_flavB2);
   fChain->SetBranchAddress("flavJ3", &flavJ3, &b_flavJ3);
   fChain->SetBranchAddress("nbJets", &nbJets, &b_nbJets);
   fChain->SetBranchAddress("isMerged", &isMerged, &b_isMerged);
   fChain->SetBranchAddress("nFatJets", &nFatJets, &b_nFatJets);
   fChain->SetBranchAddress("nFwdJets", &nFwdJets, &b_nFwdJets);
   fChain->SetBranchAddress("nSigJets", &nSigJets, &b_nSigJets);
   fChain->SetBranchAddress("nTrkJets", &nTrkJets, &b_nTrkJets);
   fChain->SetBranchAddress("RunNumber", &RunNumber, &b_RunNumber);
   fChain->SetBranchAddress("isOverlap", &isOverlap, &b_isOverlap);
   fChain->SetBranchAddress("nBTrkJets", &nBTrkJets, &b_nBTrkJets);
   fChain->SetBranchAddress("isResolved", &isResolved, &b_isResolved);
   fChain->SetBranchAddress("nbTagsInFJ", &nbTagsInFJ, &b_nbTagsInFJ);
   fChain->SetBranchAddress("passVRJetOR", &passVRJetOR, &b_passVRJetOR);
   fChain->SetBranchAddress("nTrkjetsInFJ", &nTrkjetsInFJ, &b_nTrkjetsInFJ);
   fChain->SetBranchAddress("MCChannelNumber", &MCChannelNumber, &b_MCChannelNumber);
   fChain->SetBranchAddress("nbTagsOutsideFJ", &nbTagsOutsideFJ, &b_nbTagsOutsideFJ);
   fChain->SetBranchAddress("Njets_truth_pTjet30", &Njets_truth_pTjet30, &b_Njets_truth_pTjet30);
   fChain->SetBranchAddress("IsTaggedTrkJet1InLeadFJ", &IsTaggedTrkJet1InLeadFJ, &b_IsTaggedTrkJet1InLeadFJ);
   fChain->SetBranchAddress("IsTaggedTrkJet2InLeadFJ", &IsTaggedTrkJet2InLeadFJ, &b_IsTaggedTrkJet2InLeadFJ);
   fChain->SetBranchAddress("IsTaggedTrkJet3InLeadFJ", &IsTaggedTrkJet3InLeadFJ, &b_IsTaggedTrkJet3InLeadFJ);
   fChain->SetBranchAddress("IsTaggedTrkJet1NotInLeadFJ", &IsTaggedTrkJet1NotInLeadFJ, &b_IsTaggedTrkJet1NotInLeadFJ);
//    fChain->SetBranchAddress("nFatJets", &nFatJets, &b_nFatJets);
   fChain->SetBranchAddress("NAdditionalCaloJets", &NAdditionalCaloJets, &b_NAdditionalCaloJets);
   fChain->SetBranchAddress("NMatchedTrackJetLeadFatJet", &NMatchedTrackJetLeadFatJet, &b_NMatchedTrackJetLeadFatJet);
   fChain->SetBranchAddress("NBTagMatchedTrackJetLeadFatJet", &NBTagMatchedTrackJetLeadFatJet, &b_NBTagMatchedTrackJetLeadFatJet);
   fChain->SetBranchAddress("NBTagUnmatchedTrackJetLeadFatJet", &NBTagUnmatchedTrackJetLeadFatJet, &b_NBTagUnmatchedTrackJetLeadFatJet);
   fChain->SetBranchAddress("EventNumber", &EventNumber, &b_EventNumber);
   fChain->SetBranchAddress("MET", &MET, &b_MET);
   fChain->SetBranchAddress("mB1", &mB1, &b_mB1);
   fChain->SetBranchAddress("mB2", &mB2, &b_mB2);
   fChain->SetBranchAddress("mBB", &mBB, &b_mBB);
   fChain->SetBranchAddress("mJ3", &mJ3, &b_mJ3);
   fChain->SetBranchAddress("pTV", &pTV, &b_pTV);
   fChain->SetBranchAddress("dRBB", &dRBB, &b_dRBB);
   fChain->SetBranchAddress("mBBJ", &mBBJ, &b_mBBJ);
   fChain->SetBranchAddress("pTB1", &pTB1, &b_pTB1);
   fChain->SetBranchAddress("pTB2", &pTB2, &b_pTB2);
   fChain->SetBranchAddress("pTBB", &pTBB, &b_pTBB);
   fChain->SetBranchAddress("pTJ3", &pTJ3, &b_pTJ3);
   fChain->SetBranchAddress("etaB1", &etaB1, &b_etaB1);
   fChain->SetBranchAddress("etaB2", &etaB2, &b_etaB2);
   fChain->SetBranchAddress("etaBB", &etaBB, &b_etaBB);
   fChain->SetBranchAddress("etaJ3", &etaJ3, &b_etaJ3);
   fChain->SetBranchAddress("pTBBJ", &pTBBJ, &b_pTBBJ);
   fChain->SetBranchAddress("phiB1", &phiB1, &b_phiB1);
   fChain->SetBranchAddress("phiB2", &phiB2, &b_phiB2);
   fChain->SetBranchAddress("phiBB", &phiBB, &b_phiBB);
   fChain->SetBranchAddress("phiJ3", &phiJ3, &b_phiJ3);
   fChain->SetBranchAddress("BTagSF", &BTagSF, &b_BTagSF);
   fChain->SetBranchAddress("dEtaBB", &dEtaBB, &b_dEtaBB);
   fChain->SetBranchAddress("dPhiBB", &dPhiBB, &b_dPhiBB);
   fChain->SetBranchAddress("metSig", &metSig, &b_metSig);
   fChain->SetBranchAddress("softMET", &softMET, &b_softMET);
   fChain->SetBranchAddress("ActualMu", &ActualMu, &b_ActualMu);
   fChain->SetBranchAddress("MV2c10B1", &MV2c10B1, &b_MV2c10B1);
   fChain->SetBranchAddress("MV2c10B2", &MV2c10B2, &b_MV2c10B2);
   fChain->SetBranchAddress("PUWeight", &PUWeight, &b_PUWeight);
   fChain->SetBranchAddress("mH_truth", &mH_truth, &b_mH_truth);
   fChain->SetBranchAddress("mV_truth", &mV_truth, &b_mV_truth);
   fChain->SetBranchAddress("AverageMu", &AverageMu, &b_AverageMu);
   fChain->SetBranchAddress("HTBoosted", &HTBoosted, &b_HTBoosted);
   fChain->SetBranchAddress("JVTWeight", &JVTWeight, &b_JVTWeight);
   fChain->SetBranchAddress("TriggerSF", &TriggerSF, &b_TriggerSF);
   fChain->SetBranchAddress("pTH_truth", &pTH_truth, &b_pTH_truth);
   fChain->SetBranchAddress("pTV_truth", &pTV_truth, &b_pTV_truth);
   fChain->SetBranchAddress("sumPtJets", &sumPtJets, &b_sumPtJets);
   fChain->SetBranchAddress("LumiWeight", &LumiWeight, &b_LumiWeight);
   fChain->SetBranchAddress("etaH_truth", &etaH_truth, &b_etaH_truth);
   fChain->SetBranchAddress("etaV_truth", &etaV_truth, &b_etaV_truth);
   fChain->SetBranchAddress("phiH_truth", &phiH_truth, &b_phiH_truth);
   fChain->SetBranchAddress("phiV_truth", &phiV_truth, &b_phiV_truth);
   fChain->SetBranchAddress("EventWeight", &EventWeight, &b_EventWeight);
   fChain->SetBranchAddress("bin_MV2c10B1", &bin_MV2c10B1, &b_bin_MV2c10B1);
   fChain->SetBranchAddress("bin_MV2c10B2", &bin_MV2c10B2, &b_bin_MV2c10B2);
   fChain->SetBranchAddress("bin_MV2c10J3", &bin_MV2c10J3, &b_bin_MV2c10J3);
   fChain->SetBranchAddress("MCEventWeight", &MCEventWeight, &b_MCEventWeight);
   fChain->SetBranchAddress("pTAddCaloJets", &pTAddCaloJets, &b_pTAddCaloJets);
   fChain->SetBranchAddress("ActualMuScaled", &ActualMuScaled, &b_ActualMuScaled);
   fChain->SetBranchAddress("AverageMuScaled", &AverageMuScaled, &b_AverageMuScaled);
   fChain->SetBranchAddress("C2", &C2, &b_C2);
   fChain->SetBranchAddress("D2", &D2, &b_D2);
   fChain->SetBranchAddress("HT", &HT, &b_HT);
   fChain->SetBranchAddress("mJ", &mJ, &b_mJ);
   fChain->SetBranchAddress("mVJ", &mVJ, &b_mVJ);
   fChain->SetBranchAddress("mva", &mva, &b_mva);
   fChain->SetBranchAddress("pTJ", &pTJ, &b_pTJ);
   fChain->SetBranchAddress("EtaJ", &EtaJ, &b_EtaJ);
   fChain->SetBranchAddress("PhiJ", &PhiJ, &b_PhiJ);
   fChain->SetBranchAddress("Tau21", &Tau21, &b_Tau21);
   fChain->SetBranchAddress("phiMET", &phiMET, &b_phiMET);
   fChain->SetBranchAddress("phiMPT", &phiMPT, &b_phiMPT);
   fChain->SetBranchAddress("SUSYMET", &SUSYMET, &b_SUSYMET);
   fChain->SetBranchAddress("deltaYVJ", &deltaYVJ, &b_deltaYVJ);
   fChain->SetBranchAddress("MET_Track", &MET_Track, &b_MET_Track);
   fChain->SetBranchAddress("dPhiMETMPT", &dPhiMETMPT, &b_dPhiMETMPT);
   fChain->SetBranchAddress("mvadiboson", &mvadiboson, &b_mvadiboson);
   fChain->SetBranchAddress("MindPhiMETJet", &MindPhiMETJet, &b_MindPhiMETJet);
   fChain->SetBranchAddress("absdeltaPhiVJ", &absdeltaPhiVJ, &b_absdeltaPhiVJ);
   fChain->SetBranchAddress("dEtabTrkJbTrkJ", &dEtabTrkJbTrkJ, &b_dEtabTrkJbTrkJ);
   fChain->SetBranchAddress("MTrkJet1InLeadFJ", &MTrkJet1InLeadFJ, &b_MTrkJet1InLeadFJ);
   fChain->SetBranchAddress("MTrkJet2InLeadFJ", &MTrkJet2InLeadFJ, &b_MTrkJet2InLeadFJ);
   fChain->SetBranchAddress("MTrkJet3InLeadFJ", &MTrkJet3InLeadFJ, &b_MTrkJet3InLeadFJ);
   fChain->SetBranchAddress("bin_MV2c10BTrkJ1", &bin_MV2c10BTrkJ1, &b_bin_MV2c10BTrkJ1);
   fChain->SetBranchAddress("bin_MV2c10BTrkJ2", &bin_MV2c10BTrkJ2, &b_bin_MV2c10BTrkJ2);
   fChain->SetBranchAddress("deltaRbTrkJbTrkJ", &deltaRbTrkJbTrkJ, &b_deltaRbTrkJbTrkJ);
   fChain->SetBranchAddress("PtTrkJet1InLeadFJ", &PtTrkJet1InLeadFJ, &b_PtTrkJet1InLeadFJ);
   fChain->SetBranchAddress("PtTrkJet2InLeadFJ", &PtTrkJet2InLeadFJ, &b_PtTrkJet2InLeadFJ);
   fChain->SetBranchAddress("PtTrkJet3InLeadFJ", &PtTrkJet3InLeadFJ, &b_PtTrkJet3InLeadFJ);
   fChain->SetBranchAddress("EtaTrkJet1InLeadFJ", &EtaTrkJet1InLeadFJ, &b_EtaTrkJet1InLeadFJ);
   fChain->SetBranchAddress("EtaTrkJet2InLeadFJ", &EtaTrkJet2InLeadFJ, &b_EtaTrkJet2InLeadFJ);
   fChain->SetBranchAddress("EtaTrkJet3InLeadFJ", &EtaTrkJet3InLeadFJ, &b_EtaTrkJet3InLeadFJ);
   fChain->SetBranchAddress("MV2TrkJet1InLeadFJ", &MV2TrkJet1InLeadFJ, &b_MV2TrkJet1InLeadFJ);
   fChain->SetBranchAddress("MV2TrkJet2InLeadFJ", &MV2TrkJet2InLeadFJ, &b_MV2TrkJet2InLeadFJ);
   fChain->SetBranchAddress("PhiTrkJet1InLeadFJ", &PhiTrkJet1InLeadFJ, &b_PhiTrkJet1InLeadFJ);
   fChain->SetBranchAddress("PhiTrkJet2InLeadFJ", &PhiTrkJet2InLeadFJ, &b_PhiTrkJet2InLeadFJ);
   fChain->SetBranchAddress("PhiTrkJet3InLeadFJ", &PhiTrkJet3InLeadFJ, &b_PhiTrkJet3InLeadFJ);
   fChain->SetBranchAddress("deltaPhibTrkJbTrkJ", &deltaPhibTrkJbTrkJ, &b_deltaPhibTrkJbTrkJ);
   fChain->SetBranchAddress("PtTrkJet1NotInLeadFJ", &PtTrkJet1NotInLeadFJ, &b_PtTrkJet1NotInLeadFJ);
   fChain->SetBranchAddress("dPhiMETdijetResolved", &dPhiMETdijetResolved, &b_dPhiMETdijetResolved);
   fChain->SetBranchAddress("DL1_pbTrkJet1InLeadFJ", &DL1_pbTrkJet1InLeadFJ, &b_DL1_pbTrkJet1InLeadFJ);
   fChain->SetBranchAddress("DL1_pbTrkJet2InLeadFJ", &DL1_pbTrkJet2InLeadFJ, &b_DL1_pbTrkJet2InLeadFJ);
   fChain->SetBranchAddress("DL1_pcTrkJet1InLeadFJ", &DL1_pcTrkJet1InLeadFJ, &b_DL1_pcTrkJet1InLeadFJ);
   fChain->SetBranchAddress("DL1_pcTrkJet2InLeadFJ", &DL1_pcTrkJet2InLeadFJ, &b_DL1_pcTrkJet2InLeadFJ);
   fChain->SetBranchAddress("DL1_puTrkJet1InLeadFJ", &DL1_puTrkJet1InLeadFJ, &b_DL1_puTrkJet1InLeadFJ);
   fChain->SetBranchAddress("DL1_puTrkJet2InLeadFJ", &DL1_puTrkJet2InLeadFJ, &b_DL1_puTrkJet2InLeadFJ);
   fChain->SetBranchAddress("EtaTrkJet1NotInLeadFJ", &EtaTrkJet1NotInLeadFJ, &b_EtaTrkJet1NotInLeadFJ);
   fChain->SetBranchAddress("PhiTrkJet1NotInLeadFJ", &PhiTrkJet1NotInLeadFJ, &b_PhiTrkJet1NotInLeadFJ);
   fChain->SetBranchAddress("sample", &sample, &b_sample);
   fChain->SetBranchAddress("Description", &Description, &b_Description);
   fChain->SetBranchAddress("EventFlavor", &EventFlavor, &b_EventFlavor);
   fChain->SetBranchAddress("bH_pdgid1", &bH_pdgid1, &b_bH_pdgid1);
   fChain->SetBranchAddress("jetvr_nBaGA", &jetvr_nBaGA, &b_jetvr_nBaGA);
   fChain->SetBranchAddress("jetpf_truthflav", &jetpf_truthflav, &b_jetpf_truthflav);
   fChain->SetBranchAddress("jetvr_truthflav", &jetvr_truthflav, &b_jetvr_truthflav);
   fChain->SetBranchAddress("index_Pflow_to_VR", &index_Pflow_to_VR, &b_index_Pflow_to_VR);
   fChain->SetBranchAddress("bH_m1", &bH_m1, &b_bH_m1);
   fChain->SetBranchAddress("bH_pT1", &bH_pT1, &b_bH_pT1);
   fChain->SetBranchAddress("bH_eta1", &bH_eta1, &b_bH_eta1);
   fChain->SetBranchAddress("bH_phi1", &bH_phi1, &b_bH_phi1);
   fChain->SetBranchAddress("jetpf_pt", &jetpf_pt, &b_jetpf_pt);
   fChain->SetBranchAddress("jetvr_pt", &jetvr_pt, &b_jetvr_pt);
   fChain->SetBranchAddress("jetpf_eta", &jetpf_eta, &b_jetpf_eta);
   fChain->SetBranchAddress("jetpf_phi", &jetpf_phi, &b_jetpf_phi);
   fChain->SetBranchAddress("jetvr_eta", &jetvr_eta, &b_jetvr_eta);
   fChain->SetBranchAddress("jetvr_phi", &jetvr_phi, &b_jetvr_phi);
   fChain->SetBranchAddress("jetpf_mv2c10", &jetpf_mv2c10, &b_jetpf_mv2c10);
   fChain->SetBranchAddress("jetvr_mv2c10", &jetvr_mv2c10, &b_jetvr_mv2c10);
   fChain->SetBranchAddress("jetpf_DL1r_pb", &jetpf_DL1r_pb, &b_jetpf_DL1r_pb);
   fChain->SetBranchAddress("jetpf_DL1r_pc", &jetpf_DL1r_pc, &b_jetpf_DL1r_pc);
   fChain->SetBranchAddress("jetpf_DL1r_pu", &jetpf_DL1r_pu, &b_jetpf_DL1r_pu);
   fChain->SetBranchAddress("jetvr_DL1r_pb", &jetvr_DL1r_pb, &b_jetvr_DL1r_pb);
   fChain->SetBranchAddress("jetvr_DL1r_pc", &jetvr_DL1r_pc, &b_jetvr_DL1r_pc);
   fChain->SetBranchAddress("jetvr_DL1r_pu", &jetvr_DL1r_pu, &b_jetvr_DL1r_pu);
   Notify();
}

Bool_t Reader::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}

void Reader::Show(Long64_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}
Int_t Reader::Cut(Long64_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}
#endif // #ifdef Reader_cxx
