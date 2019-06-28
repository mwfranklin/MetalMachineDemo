# MetalMachineDemo
Sample scripts from my machine learning project to differentiate metalloenzymes from metalloproteins. Because this is an ongoing project, no data has been included and the scripts present do not represent the entire workflow.

BatchGrid.sh - sample bash script for submitting multiple nearly-identical jobs using a parameter file

##CalculateSITECenters.py - the SITE center is the geometric middle of a group of 3D points in a protein defined by a cluster of metal residues

GraphFeatureDistrib.py - sample graphs made with a combination of matplotlib and seaborn

GridSearch_MLAlg_CVStrat.py - the primary script for grid-searching parameter space for the algorithms we decided to implement on this data; includes feature scaling and normalization. See also CustomModules/ExtraMLFxns.py

ProcesSinglePDBScore.py - from an early stage of generating features from the output of Rosetta (a protein design software); reads in the log file and extracts relevant information sent to STDOUT

WriteFeaturesToList.py - primary script for generating features; this primarily reads in several files for each SITE and synthesizes the information into single values. See also CustomModules/grid_tools.py, CustomModules/CoordGeomFeatures.py, and CustomModules/pocket_lining.py.


Other code relevant to this project are part of my set of custom functions, which is housed in CustomModules. In particular, PDBparser.py is my response to biopython and PDBmanip is primarily for renumbering PDBs for use with Rosetta.
