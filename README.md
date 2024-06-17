# GEMCSCTriggerPrimitivesAnalyzer
This is the tool for analyzing performance of GEMCSC Trigger Primitives [emulator](https://github.com/cms-sw/cmssw/tree/master/L1Trigger/CSCTriggerPrimitives).  
It takes as an input .root output from [GEMCSCTP Reader](https://github.com/cms-sw/cmssw/tree/master/L1Trigger/CSCTriggerPrimitives/plugins). You need to turn a reader option on before running emulator to create this file.  
# Usage
See example.  
You can run it on CERN SWAN. If you want to run it locally you need to have numpy, pandas, matplotlib and uproot.  
# Matching algorithm
The way to analyse performance of GEMCSC Trigger Primitives Emulator is to find what fraction of LCT\ALCT\CLCT are emulated correctly. To find that we "match" them. If LCT from firmware and from emulation have equal parameters values (you need to specify which ones) they form a pair and their "match" value is equal to number of this pair. If LCT has not pair, it's match value is equal to 0.