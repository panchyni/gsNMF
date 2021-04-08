## DEV NOTES ###

###########
# IMPORTS #
###########

# Math & Stats
import numpy as np
import scipy as sp
from scipy import stats

# NMF
from sklearn.decomposition import NMF
from sklearn.linear_model import LinearRegression

# Plotting
import matplotlib.pyplot as plt

# Warnings
import warnings

###########
# WARNING #
###########

# Supress warnings to avoid displaying messages about convergance
warnings.filterwarnings('ignore')

#############
# FUNCTIONS #
#############

### Section 1 --- Data Process and Analysis for A549 ###

def Process_scRNA_CSV (file_lines):
    '''
    
    Takes a the lines of a scRNA CSV file and seperate the samples
    (column 1) from the values (remaing columns)
    
    '''
    
    samples = []
    values = []
    for ln in file_lines[1:]: # Skip the header line, handle seperately if needed
        split_ln = ln.split(",")
        samples.append(split_ln[0])
        values.append([float(v) for v in split_ln[1:]])
    return samples, values
        
def Reannotate_Time(time_labels):
    '''
    
    Turn specific text labels in integer valued time in hours
    
    '''
    
    Reannot_time = []
    for v in time_labels:
        if v == '0d':
            Reannot_time.append(0)
        elif v == '8h':
            Reannot_time.append(8)
        elif v == '1d':
            Reannot_time.append(24)
        elif v == '3d':
            Reannot_time.append(72)
        elif v == '7d':
            Reannot_time.append(168)
        elif v == '8h_rm':
            Reannot_time.append(176)
        elif v == '1d_rm':
            Reannot_time.append(192)
        elif v == '3d_rm':
            Reannot_time.append(240)
    
    return Reannot_time

def Time_to_Color(time_labels):
    
    Reannot_colors = []
    for v in time_labels:
        if v == '0d':
            Reannot_colors.append("#1B7837")
        elif v == '8h':
            Reannot_colors.append("#5AAE61")
        elif v == '1d':
            Reannot_colors.append("#DEB887")
        elif v == '3d':
            Reannot_colors.append("#9970AB")
        elif v == '7d':
            Reannot_colors.append("#762A83")
        elif v == '8h_rm':
            Reannot_colors.append("#00FFFF")
        elif v == '1d_rm':
            Reannot_colors.append("#1E90FF")
        elif v == '3d_rm':
            Reannot_colors.append("#00008B")
    
    return Reannot_colors

def GetNegCoefFeature(W_mat,Reannot_time):
    """
    
    Go through the features of a W-matrix produced by NMF and compare to Reannot_time
    
    Return the feature with the most negative correlation
    
    """
    
    best_corr = 0
    best_index = 0
    for i in range(W_mat.shape[1]):
        curret_feature_corr = np.corrcoef(W_mat[:,i],Reannot_time)[0,1]
                
        if curret_feature_corr < best_corr: # Take the most negative feature
            best_corr = curret_feature_corr
            best_index = i
            
    return W_mat[:,best_index], best_corr, best_index
    
def GetPosCoefFeature(W_mat,Reannot_time):
    """
    
    Go through the features of a W-matrix produced by NMF and compare to Reannot_time
    
    Return the feature with the most postive correlation
    
    """
    
    best_corr = 0
    best_index = 0
    for i in range(W_mat.shape[1]):
        curret_feature_corr = np.corrcoef(W_mat[:,i],Reannot_time)[0,1]
                
        if curret_feature_corr > best_corr: # Take the most negative feature
            best_corr = curret_feature_corr
            best_index = i
            
    return W_mat[:,best_index], best_corr, best_index

def Get_Time_Idxs(time_labels):
    '''
    
    Turn specific text labels into a list of indexes for each label
    
    '''
    
    time_0d_idxs = []
    time_8h_idxs = []
    time_1d_idxs = []
    time_3d_idxs = []
    time_7d_idxs = []
    time_8h_rm_idxs = []
    time_1d_rm_idxs = []
    time_3d_rm_idxs = []
    for i in range(len(time_labels)):
        if time_labels[i] == '0d':
            time_0d_idxs.append(i)
        elif time_labels[i] == '8h':
            time_8h_idxs.append(i)
        elif time_labels[i] == '1d':
            time_1d_idxs.append(i)
        elif time_labels[i] == '3d':
            time_3d_idxs.append(i)
        elif time_labels[i] == '7d':
            time_7d_idxs.append(i)
        elif time_labels[i] == '8h_rm':
            time_8h_rm_idxs.append(i)
        elif time_labels[i] == '1d_rm':
            time_1d_rm_idxs.append(i)
        elif time_labels[i] == '3d_rm':
            time_3d_rm_idxs.append(i)
    
    return time_0d_idxs, time_8h_idxs, time_1d_idxs, time_3d_idxs, time_7d_idxs, time_8h_rm_idxs, time_1d_rm_idxs, time_3d_rm_idxs

def ScoreFProb(feature,index1,index2):
    '''
        Compare two population (index1,index2) from the same E-score feature
        using the Mann-Whitney U-test
        
        Note: Its important that index1 represents the EARLIER time point for E scores
        and the LATER time point for M scores
    '''
    
    U, pv = stats.mannwhitneyu([feature[j] for j in index1],[feature[j] for j in index2],alternative='greater')
    f_prob = U/(len(index1)*len(index2))

    return pv, f_prob

def ScoreFProb_Drop(feature,index1,drop_values,alternative):
    '''
        Compare two population, one using a feature + index from a fit model, one from a set
        of inferred values from a dropped time point using the Mann-Whitney U-test
        
        Note: use the alternative to control E and M scores based on wether the droped time point is earlier or later
        
    '''
    
    U, pv = stats.mannwhitneyu([feature[j] for j in index1],drop_values,alternative=alternative)
    f_prob = U/(len(index1)*len(drop_values))
    
    # Correct probability if less:
    if alternative == 'less':
        f_prob = 1 - f_prob

    return pv, f_prob

def ProcessOtherModels(data_file):
    lines = [ln.strip() for ln in open(data_file,"r").readlines()]
    Evalues = [float(ln.split("\t")[1]) for ln in lines[1:]]
    Mvalues = [float(ln.split("\t")[2]) for ln in lines[1:]]
    Time = [int(ln.split("\t")[5].strip('"')) for ln in lines[1:]]
    
    Colors  = []
    for v in Time:
        if v == 0:
            Colors.append("#1B7837")
        elif v == 8:
            Colors.append("#5AAE61")
        elif v == 24:
            Colors.append("#DEB887")
        elif v == 72:
            Colors.append("#9970AB")
        elif v == 168:
            Colors.append("#762A83")
    
    index_0d = []
    index_8h = []
    index_1d = []
    index_3d = []
    index_7d = []
    
    for i in range(len(Time)):
        v = Time[i]
        
        if v == 0:
            index_0d.append(i)
        elif v == 8:
            index_8h.append(i)
        elif v == 24:
            index_1d.append(i)
        elif v == 72:
            index_3d.append(i)
        elif v == 168:
            index_7d.append(i)
            
    Labels = [index_0d,index_8h,index_1d,index_3d,index_7d]
            
    return Evalues, Mvalues, Time, Colors, Labels

#### Section 2 --- DU145 And DU145 X A549 Models ###

def CheckSeedConsistency(EValues,Mvalues,tol=None,iter=None):

    W_E_mats = []
    W_M_mats = []

    H_E_mats = []
    H_M_mats = []

    #seeds = [1001,1002,1003]
    seeds = [1001,1002,1003,1004,1005,1006,1007,1008,1009,1010]
    for seed in seeds:

        # Make base model
        if tol == None and iter == None:
            model = NMF(n_components=2, init='random', random_state=seed)
        else:
            model = NMF(n_components=2, init='random', random_state=seed,tol=tol, max_iter=iter)

        # E data: Train model
        W_E_train = model.fit_transform(EValues)
        H_E_train = model.components_

        # M data: Train model
        W_M_train = model.fit_transform(Mvalues)
        H_M_train = model.components_

        W_E_mats.append(W_E_train)
        W_M_mats.append(W_M_train)

        H_E_mats.append(H_E_train)
        H_M_mats.append(H_M_train)

    W_E_PCC = []
    W_M_PCC = []

    H_E_PCC = []
    H_M_PCC = []
    
    for i in range(len(W_E_mats)):
        for j in range(i+1,len(W_E_mats)):

            # Compare W_E correlation
            W_E_0vs0 = np.corrcoef(W_E_mats[i][:,0],W_E_mats[j][:,0])[0,1]
            W_E_0vs1 = np.corrcoef(W_E_mats[i][:,0],W_E_mats[j][:,1])[0,1]
            W_E_1vs0 = np.corrcoef(W_E_mats[i][:,1],W_E_mats[j][:,0])[0,1]
            W_E_1vs1 = np.corrcoef(W_E_mats[i][:,1],W_E_mats[j][:,1])[0,1]

            # Save best
            W_E_PCC.append(max(W_E_0vs0,W_E_0vs1))
            W_E_PCC.append(max(W_E_1vs0,W_E_1vs1))

            # Compare W_M correlation
            W_M_0vs0 = np.corrcoef(W_M_mats[i][:,0],W_M_mats[j][:,0])[0,1]
            W_M_0vs1 = np.corrcoef(W_M_mats[i][:,0],W_M_mats[j][:,1])[0,1]
            W_M_1vs0 = np.corrcoef(W_M_mats[i][:,1],W_M_mats[j][:,0])[0,1]
            W_M_1vs1 = np.corrcoef(W_M_mats[i][:,1],W_M_mats[j][:,1])[0,1]

            # Save best
            W_M_PCC.append(max(W_M_0vs0,W_M_0vs1))
            W_M_PCC.append(max(W_M_1vs0,W_M_1vs1))

            # Compare H_E correlation
            H_E_0vs0 = np.corrcoef(H_E_mats[i][0,:],H_E_mats[j][0,:])[0,1]
            H_E_0vs1 = np.corrcoef(H_E_mats[i][0,:],H_E_mats[j][1,:])[0,1]
            H_E_1vs0 = np.corrcoef(H_E_mats[i][1,:],H_E_mats[j][0,:])[0,1]
            H_E_1vs1 = np.corrcoef(H_E_mats[i][1,:],H_E_mats[j][1,:])[0,1]

            # Save best
            H_E_PCC.append(max(H_E_0vs0,H_E_0vs1))
            H_E_PCC.append(max(H_E_1vs0,H_E_1vs1))

            # Compare H_E correlation
            H_M_0vs0 = np.corrcoef(H_M_mats[i][0,:],H_M_mats[j][0,:])[0,1]
            H_M_0vs1 = np.corrcoef(H_M_mats[i][0,:],H_M_mats[j][1,:])[0,1]
            H_M_1vs0 = np.corrcoef(H_M_mats[i][1,:],H_M_mats[j][0,:])[0,1]
            H_M_1vs1 = np.corrcoef(H_M_mats[i][1,:],H_M_mats[j][1,:])[0,1]

            # Save best
            H_M_PCC.append(max(H_M_0vs0,H_M_0vs1))
            H_M_PCC.append(max(H_M_1vs0,H_M_1vs1))
            
    #return W_E_PCC, W_M_PCC, H_E_PCC, H_M_PCC
    return W_E_PCC, W_M_PCC

def FilterDataByGenes_scRNA(Data_A,Genes_A,Data_B,Genes_B):
    
    # Transpose Data to make genes values along rows
    Data_A_transpose = np.transpose(Data_A)
    Data_B_transpose = np.transpose(Data_B)
    
    # Store rows according to genes
    Data_A_dict = {}

    for i in range(len(Genes_A)):
        Data_A_dict[Genes_A[i]] = Data_A_transpose[i,]
    
    Data_B_dict = {}
    for i in range(len(Genes_B)):
        Data_B_dict[Genes_B[i]] = Data_B_transpose[i,]
        
    # Filter by common genes:
    filter_values_A = []
    filter_values_B = []
    common_genes = []
    for gene in Genes_B:
        
        # Note: using if is slower than finding an intersection, but it helps preserver order of genes
        if gene in Data_A_dict.keys() and Data_B_dict.keys():
            filter_values_A.append(Data_A_dict[gene])
            filter_values_B.append(Data_B_dict[gene])
            common_genes.append(gene)
            
    filter_values_A_transpose = np.transpose(filter_values_A)
    filter_values_B_transpose = np.transpose(filter_values_B)
    
    return filter_values_A_transpose, filter_values_B_transpose, common_genes

### Section 3 --- TCGA Data ###
def FilterDataByGenes_TCGA(Data_scRNA,Genes_scRNA,TCGA_dict):
    
    # Transpose scRNA Data to make genes values along rows
    Data_scRNA_transpose = np.transpose(Data_scRNA)

    # Store rows according to genes
    Data_scRNA_dict = {}

    for i in range(len(Genes_scRNA)):
        Data_scRNA_dict[Genes_scRNA[i]] = Data_scRNA_transpose[i,]
        
    # Filter by common genes:
    filter_values_scRNA = []
    filter_values_TCGA = []
    for gene in Genes_scRNA:
        
        # Note: using if is slower than finding an intersection, but it helps preserver order of genes
        if gene in Data_scRNA_dict.keys() and gene in TCGA_dict.keys():
            filter_values_scRNA.append(Data_scRNA_dict[gene])
            filter_values_TCGA.append(TCGA_dict[gene])
            
    filter_values_scRNA_transpose = np.transpose(filter_values_scRNA)
    filter_values_TCGA_transpose = np.transpose(filter_values_TCGA)
    
    return filter_values_scRNA_transpose, filter_values_TCGA_transpose

def TCGAdata_to_dict(TCGA_lines):
    '''

    Convert a list of lines for a TCGA file into a dictionary of lines
    keyed by gene

    '''

    TCGA_dict = {}
    for ln in TCGA_lines[1:]:
    	split_ln = ln.split(",")
    	name = split_ln[0]
    	values = split_ln[1:]
    	TCGA_dict[name] = values

    return TCGA_dict

def AlignScores (TCGA_samples, TCGA_dict, A549_Escores, A549_Mscores, DU145_Escores, DU145_Mscores):
    A549_Escore = []
    A549_Mscore = []
    DU145_Escore = []
    DU145_Mscore = []
    previous_Escore = []
    previous_Mscore = []

    for i in range(len(TCGA_samples)):
        TCGA_name = TCGA_samples[i]

        if TCGA_name in TCGA_dict.keys():
            A549_Escore.append(A549_Escores[i])
            A549_Mscore.append(A549_Mscores[i])
            DU145_Escore.append(DU145_Escores[i])
            DU145_Mscore.append(DU145_Mscores[i])
            previous_Escore.append(TCGA_dict[TCGA_name][0])
            previous_Mscore.append(TCGA_dict[TCGA_name][1])
            
    return A549_Escore, A549_Mscore, DU145_Escore, DU145_Mscore, previous_Escore, previous_Mscore

def PreviousValues_Lines2Dict (prev_value_lines):
    '''

    Convert lines from a previous values file into a dictionary
    keyed by samples

    '''

    prev_values_dict = {}
    for ln in prev_value_lines[1:]:
        split_ln = ln.split(" ")
        name = "_".join(split_ln[0].strip('"').strip("'").split("_")[2:])
        prev_values_dict[name] = [float(split_ln[1]),float(split_ln[2])]

    return prev_values_dict

### Section 4 --- Mock X TGFB 

def Position_to_Color(posit_labels):
    
    Reannot_colors = []
    for v in posit_labels:
        if v == 'inner':
            Reannot_colors.append("#D30000")
        elif v == 'outer':
            Reannot_colors.append("#0018F9")
    
    return Reannot_colors

def Best_EFeature_InOut(E_feature,in_idxs,out_idxs):
    dim0_stat, dim0_pv = sp.stats.mannwhitneyu(E_feature[in_idxs,0],E_feature[out_idxs,0],alternative='greater')
    f0_prob = dim0_stat/(len(in_idxs)*len(out_idxs))
    
    dim1_stat, dim1_pv = sp.stats.mannwhitneyu(E_feature[in_idxs,1],E_feature[out_idxs,1],alternative='greater')
    f1_prob = dim1_stat/(len(in_idxs)*len(out_idxs))
    
    if f0_prob > f1_prob:
        return E_feature[:,0], dim0_pv, f0_prob, 0
    else:
        return E_feature[:,1], dim1_pv, f1_prob, 1

def Best_MFeature_InOut(M_feature,in_idxs,out_idxs):
    dim0_stat, dim0_pv = sp.stats.mannwhitneyu(M_feature[in_idxs,0],M_feature[out_idxs,0],alternative='less')
    f0_prob = dim0_stat/(len(in_idxs)*len(out_idxs))
    f0_prob = 1 - f0_prob
    
    dim1_stat, dim1_pv = sp.stats.mannwhitneyu(M_feature[in_idxs,1],M_feature[out_idxs,1],alternative='less')
    f1_prob = dim1_stat/(len(in_idxs)*len(out_idxs))
    f1_prob = 1 - f1_prob
    
    if f0_prob > f1_prob:
        return M_feature[:,0], dim0_pv, f0_prob, 0
    else:
        return M_feature[:,1], dim1_pv, f1_prob, 1

def GetGenes(dictionary,gene_set,threshold):
    
    return_genes = []
    for key in dictionary.keys():
        if dictionary[key][gene_set][0] >= threshold:
            return_genes.append(key)
            
    return return_genes

def GetGenes_Down(dictionary,gene_set,threshold):
    
    return_genes = []
    for key in dictionary.keys():
        if dictionary[key][gene_set][0] <= threshold:
            return_genes.append(key)
            
    return return_genes

def MergeDictionaries(spatial_dict,A549_dict):
    
    combined_dict  = {}
    for name in spatial_dict.keys():
        if name in A549_dict.keys():
            combined_dict[name] = {"Base": [], "A549": []}
            combined_dict[name]["Base"].append(spatial_dict[name])
            combined_dict[name]["A549"].append(A549_dict[name])
    
    return combined_dict

### Section --- C2 Clustering ###
def Filter_C2Keys(NMF_results_dict):
    
    key_var_values = []
    key_var_mean_values = []
    filtered_keys = []
    for key in NMF_results_dict[1001].keys():
    
        key_R_values = []
        for seed in [1001,1002,1003,1004,1005,1006,1007,1008,1009,1010]:
            key_R_values.append(NMF_results_dict[seed][key]["no_Rev_Best_R"])

        key_mean = np.mean([abs(v) for v in key_R_values])
        key_var = np.var(key_R_values)

        key_var_values.append(key_var)
        key_var_mean_values.append(key_var/abs(key_mean))

        if key_var/key_mean < 0.05:
            filtered_keys.append(key)
            
    return filtered_keys

def AverageSeedPCCs(filtered_keys,NMF_results_dict):

    module_seedMEAN_PCC_dict = {}
    
    for key in filtered_keys:
        seed_values = []
        for seed in [1001,1002,1003,1004,1005,1006,1007,1008,1009,1010]:
            seed_values.append(NMF_results_dict[seed][key]["no_Rev_Best_R"])
        module_seedMEAN_PCC_dict[key] = np.mean(seed_values)
    
    return module_seedMEAN_PCC_dict

def SeedR2Correlation(NMF_results_dict,keys):

    module_results_byseed = []
    for seed in [1001,1002,1003,1004,1005,1006,1007,1008,1009,1010]:
        module_results_byseed.append([NMF_results_dict[seed][key]["no_Rev_Best_R"] for key in keys])

    SeedCorrelationMatrix = np.corrcoef(module_results_byseed)
    upper_tri_PCCvalues = []
    for i in range(10):
        for j in range (i+1,10):
            upper_tri_PCCvalues.append(SeedCorrelationMatrix[i,j])
    
    return upper_tri_PCCvalues

### Section 6 --- Denovo Gene Moudles

def Create_GeneValues_Dict (module_genes_dict,H_dict,results_dict,filter_keys,All_genes):
    '''
    
    Makes a dictionary of genes and values for denovo clustering
    
    '''

    # Make Intermediate dictionary for averaging across seeds and clsuters
    sum_values_dict = {}
    for key in filter_keys:

        gene_list = module_genes_dict[1001][key]
        overlap = list(set(All_genes).intersection(set(gene_list)))
        overlapping_genes = [All_genes[i] for i in range(len(All_genes)) if All_genes[i] in overlap]

        for seed in [1001,1002,1003,1004,1005,1006,1007,1008,1009,1010]:

            #print(key + "\t" + str(seed))
            H_matrix = H_dict[seed][key]
            results = results_dict[seed][key]

            PC_index = -1
            if results['no_Rev_Best_R'] == results['no_Rev_PC1_R']:
                PC_index = 0
            else:
                PC_index = 1

            for i in range(len(overlapping_genes)):

                gene = overlapping_genes[i]
                H_value = H_matrix[PC_index,i]
                no_rev_best = results['no_Rev_Best_R']

                if gene in sum_values_dict.keys():
                    sum_values_dict[gene]["Count"] = sum_values_dict[gene]["Count"] + 1
                    sum_values_dict[gene]["H_sum"] = sum_values_dict[gene]["H_sum"] + H_value
                    sum_values_dict[gene]["Corr_sum"] = sum_values_dict[gene]["Corr_sum"] + no_rev_best
                else:
                    sum_values_dict[gene] = {"Count": 1, "H_sum": H_value, "Corr_sum": no_rev_best}
    
    # Average Finals values
    gene_values_dict = {}
    for gene in sum_values_dict.keys():
        gene_values_dict[gene] = {}
        gene_values_dict[gene]["Count"] = sum_values_dict[gene]["Count"]/10 # Account for seeds
        gene_values_dict[gene]["H_mean"] = sum_values_dict[gene]["H_sum"]/sum_values_dict[gene]["Count"]
        gene_values_dict[gene]["Corr_mean"] = sum_values_dict[gene]["Corr_sum"]/sum_values_dict[gene]["Count"]
        
    return gene_values_dict

def GetDenovoModules(gene_values_dict,gene_keys):
    
    min_corr_values = [0.0,0.1,0.2,0.3,0.4,0.5]
    percentile_values = [80,85,90,95,98,99,99.5]
    
    H_values = []
    for gene in gene_keys:
        H_values.append(gene_values_dict[gene]["H_mean"])

    gene_module_dict = {}
    for min_corr in min_corr_values:
    
        gene_module_dict[min_corr] = {}
        for percentile in percentile_values:

            h_percent = np.percentile(H_values,percentile)
            gene_module_dict[min_corr][percentile] = []
            for gene in gene_keys:
            
                if gene_values_dict[gene]["H_mean"] >= h_percent and gene_values_dict[gene]["Corr_mean"] >= min_corr:
                    gene_module_dict[min_corr][percentile].append(gene)
    return gene_module_dict

def EvaluateModules(gene_modules_dict,All_values,All_genes,Time_values):
    min_corr_values = [0.0,0.1,0.2,0.3,0.4,0.5]
    percentile_values = [80,85,90,95,98,99,99.5]
    out_dict = {}

    for min_corr in min_corr_values:

        out_dict[min_corr] = {}
        for percentile in percentile_values:

            out_dict[min_corr][percentile] = []
            gene_list = gene_modules_dict[min_corr][percentile]

            for seed in [1001,1002,1003,1004,1005,1006,1007,1008,1009,1010]:

                # Filter
                ModuleScale_values = [All_values[i] for i in range(len(All_genes)) if All_genes[i] in gene_list]
                ModuleScale_values_T = np.transpose(np.array(ModuleScale_values))

                # Model
                model = NMF(n_components=2, init='random', random_state=seed,tol=1e-9, max_iter=2500)

                # Train
                W_module = model.fit_transform(ModuleScale_values_T-np.amin(ModuleScale_values_T))
                H_module = model.components_

                # Evaluate model
                PC1_corr = np.corrcoef(W_module[:,0],Time_values)[0,1]
                PC2_corr = np.corrcoef(W_module[:,1],Time_values)[0,1]

                Best_corr = 0
                if abs(PC1_corr) > abs(PC2_corr):
                    Best_corr = PC1_corr
                else:
                    Best_corr = PC2_corr


		### Evaluate without revertant points

                # Get indexes for non-revertant samples
                no_rev_index = []
                for i in range(len(Time_values)):
                    if Time_values[i] < 176:
                        no_rev_index.append(i)
        
                no_rev_Time = [Time_values[i] for i in no_rev_index]
                W_no_rev_PC1 = [W_module[i,0] for i in no_rev_index]
                W_no_rev_PC2 = [W_module[i,1] for i in no_rev_index]

                PC1_no_rev_corr = np.corrcoef(W_no_rev_PC1,no_rev_Time)[0,1]
                PC2_no_rev_corr = np.corrcoef(W_no_rev_PC2,no_rev_Time)[0,1]

                Best_no_rev_corr = 0
                if abs(PC1_no_rev_corr) > abs(PC2_no_rev_corr):
                    Best_no_rev_corr = PC1_no_rev_corr
                else:
                    Best_no_rev_corr = PC2_no_rev_corr

                out_dict[min_corr][percentile].append(Best_no_rev_corr)
                
    return out_dict

### MAIN ####
if __name__ == "__main__":
  run()
