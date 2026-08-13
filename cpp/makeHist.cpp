#include <fstream>
#include <sstream>
#include <string>
#include <array>
#include <iostream>
#include <vector>
#include <map>
#include <random>
#include <algorithm>

#include "TH2F.h"
#include "TH1F.h"
#include "TCanvas.h"
#include "TAxis.h"
#include "TFile.h"
#include "TTree.h"
#include "TCanvas.h"

struct particleHit {
    double x;
    double y;
    int detectorIndex;
};

void makeHist() {
    std::ifstream infile("/Users/jayshirlee/refBeam_179.txt"); // text file with data from p0 = 179 MeV/c
    
    if (infile.fail()) { // if file isn't accessible
        std::cout << "Error: could not open file" << std::endl;
    }
    
    //STL structures
    std::string line;
    std::map<int, std::vector<particleHit>> eventsMap; // struct for particle RNG
    std::array<int, 10> zPositions = {0, 3575, 6435, 9295, 12155, 15015, 17875, 20735, 23595, 26455};
    std::random_device rng;
    std::mt19937 g(rng());
    
    double x=0, y=0, z=0, Px=0, Py=0, Pz=0, t=0, Weight=0;
    int PDGid=0, EventID=0, TrackID=0, ParentID=0;
    
    // Loop 2: Parsing each line in data file to fill histograms
    while (getline(infile, line)) {
        // Skip comments/header lines
        if (line.empty()) {
            continue;
        }
        if (line[0] == '#') {
            continue;
        }
        
        std::stringstream ss(line);
        
        ss >> x >> y >> z
        >> Px >> Py >> Pz
        >> t
        >> PDGid >> EventID >> TrackID >> ParentID
        >> Weight;
        
        for (int i = 0; i < zPositions.size(); i++) {
            if (z == zPositions[i]) {
                // Filling 2D histograms
                //hXpx->Fill(i,x);
                particleHit hit;
                hit.x = x;
                hit.y = y;
                hit.detectorIndex = i;
                eventsMap[EventID].push_back(hit);
            }
        }
    }
    
    // RNG creation
    std::vector<int> allEventIDs; // declaring vector for all respective event IDS
    for (std::map<int, std::vector<particleHit>>::iterator iter = eventsMap.begin(); iter != eventsMap.end(); ++iter) {
        // for loop that creates iterator that picks a unique event in a a randomized set
        allEventIDs.push_back(iter->first);
    }
    
    for (int j = 0; j < 100; j++) {
        // shuffling list of all respective ID's
        std::shuffle(allEventIDs.begin(), allEventIDs.end(), g);
        
        // 2D Histograms
        std::string histName = "hZx_" + std::to_string(j);
        TH2F *hXpx = new TH2F(histName.c_str() , "X Position vs. Virtual Detector Z Location", 10, -0.5, 9.5, 10, -220, 220);
        
        
        // 3. Take the first 100 unique events from the shuffled list
        for (int k = 0; k < 100; k++) {
            int selectedEventID = allEventIDs[k]; // no duplicate check
            
            // get all hits associated with this selected event from map
            std::vector<particleHit> hits = eventsMap[selectedEventID];
            for (int h = 0; h < hits.size(); h++) {
                hXpx->Fill(hits[h].detectorIndex, hits[h].x);
            }
   
        }
        
        for (int l = 0; l < zPositions.size(); l++) {
            std::string binlabel = std::to_string(zPositions[l]);
            hXpx->GetXaxis()->SetBinLabel(l + 1, binlabel.c_str());
        }
        
        // canvas object
        TCanvas *c1 = new TCanvas("c1", "Histograms", 1200, 400);
        c1->Divide(1, 1);
        
        // 2D Histogram creation
        c1->cd(1);
        hXpx->SetStats(0);
        hXpx->GetYaxis()->SetTitle("X position  [mm] ");
        hXpx->GetXaxis()->SetTitle("Z position [mm]");
        hXpx->GetZaxis()->SetTitle("Events");
        hXpx->Draw("COLZ");
        
        // saving files
        std::string savePath = "/Users/jayshirlee/Library/CloudStorage/GoogleDrive-jayeushirlee06@gmail.com/.shortcut-targets-by-id/1pfoqvybLTublKRj2oJtsiybF4sL1U5bO/AI/179_S/2DBHisto179_100_" + std::to_string(j) + ".png";
        c1->SaveAs(savePath.c_str());
        
        delete hXpx;
        delete c1;
    }
}
    

int main() {
    makeHist();
    return 0;
}
