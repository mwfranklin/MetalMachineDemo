import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.lines import Line2D
import seaborn as sns
from scipy import interpolate
from scipy import stats
import pandas as pd
import numpy as np
import math
from collections import Counter
from sklearn import preprocessing, cluster
from sklearn.impute import SimpleImputer
import ExtraMLFxns as mlf #this is housed in CustomModules

def graph_kdes(names, scores, filename):
    n_rows = math.ceil(len(names)/5)
    fig, axes = plt.subplots(nrows = n_rows, ncols= 5, figsize = (n_rows*8,25))
    
    for x in range(0, len(names)):
        #print(names[x])
        these_scores = scores.dropna(subset = [names[x]])
        this_axes = axes[x//5, x%5]
        sns.kdeplot( these_scores[names[x]][these_scores.Catalytic == True], shade=True, color = "green", ax=this_axes, cut = 0 )
        sns.kdeplot( these_scores[names[x]][these_scores.Catalytic == False], shade=True, color = "red", ax=this_axes, cut = 0 )
        this_axes.set_title(names[x], size = 14)   
        
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

relaxed = "Relaxed"
wp_scores = pd.read_csv("WholeScores_%s.txt"%relaxed, sep = "\t", header = 0)
wp_scores.dropna(subset=["SITE_ID"], inplace=True)
#group_sizes = wp_scores.groupby(["ValidSet", "Catalytic"]).size()
#print(group_sizes)
valid_set = wp_scores[wp_scores.ValidSet == True]
wp_scores = wp_scores[wp_scores.ValidSet == False]
bad_terms = ("hbond_lr_", 'dslf_fa13', 'pro_close')
wp_scores.drop(columns = [term for term in wp_scores if term.startswith(bad_terms)], inplace = True)
#wp_scores.describe().to_csv("FeatureGraphs/SummaryStats_%s.txt"%relaxed, sep = "\t", header = True, index = True)

gen_set = ['MetalCodes', 'MetalAtoms', 'Depth', 'Vol', "SITEDistCenter", "SITEDistNormCenter"]
gen_terms = ("BSA", 'expHP', 'LoopDSSP', 'HelixDSSP', 'SheetDSSP')
all_gen_set = [ term for term in wp_scores if term.startswith(gen_terms) ]
gen_shell = [name for name in all_gen_set if "_S" in name]
gen_sph = list(set(all_gen_set).difference(gen_shell))
gen_shell += gen_set
gen_sph += gen_set
all_gen_set += gen_set
#Rosetta terms only
ros_sum_sph0 = list(set([name for name in wp_scores if name.endswith("_Sum_3.5")]).difference(all_gen_set))
ros_sum_sph1 = list(set([ name for name in wp_scores if name.endswith("_Sum_5") ]).difference(all_gen_set))
ros_sum_sph2 = list(set([ name for name in wp_scores if name.endswith("_Sum_7.5") ]).difference(all_gen_set))
ros_sum_sph3 = list(set([ name for name in wp_scores if name.endswith("_Sum_10") ]).difference(all_gen_set))
ros_sum_shell1 = list(set([ name for name in wp_scores if name.endswith("_Sum_S5") ]).difference(all_gen_set))
ros_sum_shell2 = list(set([ name for name in wp_scores if name.endswith("_Sum_S7.5") ]).difference(all_gen_set))
ros_sum_shell3 = list(set([ name for name in wp_scores if name.endswith("_Sum_S10") ]).difference(all_gen_set))

ros_mean_sph0 = list(set([name for name in wp_scores if name.endswith("_Mean_3.5")]).difference(all_gen_set))
ros_mean_sph1 = list(set([ name for name in wp_scores if name.endswith("_Mean_5") ]).difference(all_gen_set))
ros_mean_sph2 = list(set([ name for name in wp_scores if name.endswith("_Mean_7.5") ]).difference(all_gen_set))
ros_mean_sph3 = list(set([ name for name in wp_scores if name.endswith("_Mean_10") ]).difference(all_gen_set))
ros_mean_shell1 = list(set([ name for name in wp_scores if name.endswith("_Mean_S5") ]).difference(all_gen_set))
ros_mean_shell2 = list(set([ name for name in wp_scores if name.endswith("_Mean_S7.5") ]).difference(all_gen_set))
ros_mean_shell3 = list(set([ name for name in wp_scores if name.endswith("_Mean_S10") ]).difference(all_gen_set))

electro = [name for name in wp_scores if name.startswith("Elec")]
geom = [name for name in wp_scores if name.startswith("geom")]
findgeo_geoms = ("lin", "trv", "tri", "tev", "spv", 
    "tet", "spl", "bva", "bvp", "pyv", 
    "spy", "tbp", "tpv", 
    "oct", "tpr", "pva", "pvp", "cof", "con", "ctf", "ctn",
    "pbp", "coc", "ctp", "hva", "hvp", "cuv", "sav",
    "hbp", "cub", "sqa", "boc", "bts", "btt", 
    "ttp", "csa")
geom_nocoord = [name for name in geom if not name.endswith(findgeo_geoms)]
#pocket features only
pocket_set = ['MetalCodes', 'MetalAtoms', 'SEPocket', 'Depth', 'Vol', "SITEDistCenter", "SITEDistNormCenter", 'LongPath', 'farPtLow', 'PocketAreaLow', 'OffsetLow', 'LongAxLow', 'ShortAxLow', 'farPtMid', 'PocketAreaMid', 'OffsetMid', 'LongAxMid', 'ShortAxMid', 'farPtHigh', 'PocketAreaHigh', 'OffsetHigh', 'LongAxHigh', 'ShortAxHigh']
#pocket lining only
lining_set = ['num_pocket_bb', 'num_pocket_sc', 'avg_eisen_hp', 'min_eisen', 'max_eisen', 'skew_eisen', 'std_dev_eisen', 'avg_kyte_hp', 'min_kyte', 'max_kyte', 'skew_kyte', 'std_dev_kyte', 'occ_vol', 'NoSC_vol', 'SC_vol_perc']

#print(len(lining_set), len(pocket_set), len(geom), len(electro), len(ros_sum_sph0), len(ros_sum_sph1), len(gen_shell))

#quick and dirty kde of all data points for fast visualization
graph_kdes(sorted(all_gen_set), wp_scores, "FeatureGraphs/GeneralPropertiesKDE_%s.png"%relaxed)
graph_kdes(sorted(pocket_set), wp_scores, "FeatureGraphs/PocketKDE_%s.png"%relaxed)
graph_kdes(sorted(lining_set), wp_scores, "FeatureGraphs/LiningKDE_%s.png"%relaxed)
graph_kdes(sorted(electro), wp_scores, "FeatureGraphs/ElectrostaticsKDE_%s.png"%relaxed)
graph_kdes(sorted(geom), wp_scores, "FeatureGraphs/GeomKDE_%s.png"%relaxed)
graph_kdes(sorted(geom_nocoord), wp_scores, "FeatureGraphs/GeomKDE_NoCoord_%s.png"%relaxed)


#individual pretty graphs; slightly too complicated to make it worth writing a function
#Depth, Vol, UnRelaxedBFact
fig,ax1= plt.subplots(nrows = 1, ncols = 1, figsize = (4,4))
ax1.set_xlabel(r"Pocket Depth ($\AA$)", size = 22)
ax1.set_ylabel("Density", size = 22)
xnew = np.arange(0, 650, 10)
kde = stats.gaussian_kde(wp_scores["Depth"].loc[wp_scores["Catalytic"] == True])
ax1.plot(xnew, kde.evaluate(xnew), linewidth = 2, color="#67a96c")
ax1.scatter(wp_scores["Depth"].loc[wp_scores["Catalytic"] == True], np.full(len(wp_scores.loc[wp_scores["Catalytic"] == True]), -0.001), c = "#67a96c", s = 20)
kde = stats.gaussian_kde(wp_scores["Depth"].loc[wp_scores["Catalytic"] == False])
ax1.plot(xnew, kde.evaluate(xnew), linewidth = 2, color="#3a5ca0")
ax1.scatter(wp_scores["Depth"].loc[wp_scores["Catalytic"] == False], np.full(len(wp_scores.loc[wp_scores["Catalytic"] == False]), -0.003), c = "#3a5ca0", s = 20)
ax1.set_xlim([0,250])
ax1.set_ylim([-0.005,0.07])
ax1.tick_params(labelsize=16)
plt.tight_layout()
plt.savefig("FeatureGraphs/Depth_%s.png"%relaxed)

fig,ax2= plt.subplots(nrows = 1, ncols = 1, figsize = (4.5,4))
ax2.set_xlabel(r"Pocket Volume ($\AA$$^{3}$)", size = 22)
ax2.set_ylabel("Density", size = 22)
xnew = np.arange(0, 9000, 20) #10K for unrelaxed, 9K for relaxed
kde = stats.gaussian_kde(wp_scores["Vol"].loc[wp_scores["Catalytic"] == True])
ax2.plot(xnew, kde.evaluate(xnew), linewidth = 2, color = "#67a96c")
ax2.scatter(wp_scores["Vol"].loc[wp_scores["Catalytic"] == True], np.full(len(wp_scores.loc[wp_scores["Catalytic"] == True]), -0.00005), c = "#67a96c", s = 20, label="Catalytic")
kde = stats.gaussian_kde(wp_scores["Vol"].loc[wp_scores["Catalytic"] == False])
ax2.plot(xnew, kde.evaluate(xnew), linewidth = 2, color="#3a5ca0")
ax2.scatter(wp_scores["Vol"].loc[wp_scores["Catalytic"] == False], np.full(len(wp_scores.loc[wp_scores["Catalytic"] == False]), -0.00015), c = "#3a5ca0", s = 20, label="Not Catalytic")
#y_labels = list(ax2.get_yticklabels())
#ax2.set_yticklabels(np.arange(-5e-4,3e-3,5e-4))
#ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
ax2.set_xlim([0,9000])
ax2.set_ylim([-0.0002,2.6e-3])
ax2.tick_params(labelsize=16)
plt.tight_layout()
plt.savefig("FeatureGraphs/Vol_%s.png"%relaxed)

fig,ax3= plt.subplots(nrows = 1, ncols = 1, figsize = (4,4.5))
bfact_data = pd.read_csv("AvgBFact_3.3A_UnRelaxed.txt", sep = "\t", header = 0)
unrelax_sites = pd.read_csv("SITECenters_UnRelaxed.txt", sep = "\t", header = 0, usecols=["SITE_ID", "catalytic", "ValidSet"])
bfact_data = pd.merge(bfact_data, unrelax_sites, on = "SITE_ID")
bfact_data.rename(columns = {"catalytic":"Catalytic"}, inplace= True)
bfact_data.dropna(inplace=True)
ax3.set_xlabel("Average Normalized\nB-factor", size = 22)
ax3.set_ylabel("Density", size = 22)
xnew = np.arange(-2, 5, 0.1)
kde = stats.gaussian_kde(bfact_data["AvgBFact"].loc[bfact_data["Catalytic"] == True])
ax3.plot(xnew, kde.evaluate(xnew), linewidth = 2, color = "#67a96c")
ax3.scatter(bfact_data["AvgBFact"].loc[bfact_data["Catalytic"] == True], np.full(len(bfact_data.loc[bfact_data["Catalytic"] == True]), -0.02), c = "#67a96c", s = 20, label="Catalytic")
kde = stats.gaussian_kde(bfact_data["AvgBFact"].loc[bfact_data["Catalytic"] == False])
ax3.plot(xnew, kde.evaluate(xnew), linewidth = 2, color="#3a5ca0")
ax3.scatter(bfact_data["AvgBFact"].loc[bfact_data["Catalytic"] == False], np.full(len(bfact_data.loc[bfact_data["Catalytic"] == False]), -0.06), c = "#3a5ca0", s = 20, label="Not Catalytic")
ax3.set_xlim([-2,5])
ax3.set_ylim([-0.08, 1.1])
#y_lim = list(ax3.get_ybound())
#ax3.legend(fontsize = 18, markerscale = 2)
ax3.tick_params(labelsize=16)
plt.tight_layout()
plt.savefig("FeatureGraphs/BFact_%s.png"%relaxed)