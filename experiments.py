import numpy as np
import clustering_methods as clustering
import clustering_metrics as metrics
import ds

def DSP_experiment(df, X_data, Y_data, V_data, Z_Clus, real_labels, iteration_number):

    dsp_c = []
    dsp_p = []
    dsp_f1 = []

    for i in range(iteration_number):
        #run DS+
        n_clus, labels, data = clustering.DSP_test(X_data, Y_data, V_data, Z_Clus);
        
        tp, fp, tn, fn = metrics.get_match_data(labels, real_labels)

        c = metrics.get_completitud(tp, tp + fn)
        dsp_c.append(c)
        p = metrics.get_purity(tp, tp + fp)
        dsp_p.append(p)
        dsp_f1.append(metrics.get_f1_score(c, p))
        
    return dsp_c, dsp_p, dsp_f1

def DS_experiment(df, X_data, Y_data, V_data, galaxies_in_group, iteration_number):

    ds_p_c = []
    ds_p_p = []
    ds_p_f1 = []

    
    ds_s_c = []
    ds_s_p = []
    ds_s_f1 = []

    ds_deltas = clustering.run_DS_test(X_data, Y_data, V_data)

    #print("deltas: ",ds_deltas)

    for i in range(iteration_number):

        mc_deltas = ds.monte_carlo_simulation(X_data, Y_data, V_data)


        outliers_by_percent, p_threshold = ds.find_outliers_by_percentage(ds_deltas, mc_deltas)
        outliers_by_sigma = ds.find_outliers_by_sigma(ds_deltas, mc_deltas)[0]


        #print("outliers: ",outliers_by_percent)
        #print("outliers2: ",outliers_by_sigma)

        ds_p_matches = find_ds_matches(outliers_by_percent, df)
        ds_s_matches = find_ds_matches(outliers_by_sigma, df)

        c = metrics.get_completitud(ds_p_matches, galaxies_in_group)
        p = metrics.get_purity(ds_p_matches, len(outliers_by_percent))
        ds_p_c.append(c)
        ds_p_p.append(p)
        ds_p_f1.append(metrics.get_f1_score(c, p))

        c = metrics.get_completitud(ds_s_matches, galaxies_in_group)
        p = metrics.get_purity(ds_s_matches, len(outliers_by_sigma))
        ds_s_c.append(c)
        ds_s_p.append(p)
        ds_s_f1.append(metrics.get_f1_score(c, p))

    return ds_p_c, ds_p_p, ds_p_f1, ds_s_c, ds_s_p, ds_s_f1  

def GMM_experiment(df, X_data, Y_data, Z_data, galaxies_in_group, iteration_number):
    
    n_clusters, labels, gmm_data = GMM(X_data, Y_data, Z_data)

    # Find the most repeated label
    most_repeated_label = np.bincount(labels).argmax()
    # Replace all occurrences of the most repeated label with -1
    labels[labels == most_repeated_label] = -1
    unique_labels, labels = np.unique(labels, return_inverse=True)

def get_experiment_df(samples, completitudes, purities, f1_scores):

    #df_sample_indexes = [dfs.index(df) for df in samples]
    df_sample_indexes = [i for df_sample in samples for i, df in enumerate(dfs) if df.equals(df_sample)]


    completitude_means = []
    completitude_std = []

    purity_means = []
    purity_std = []

    f1_scores_means = []
    f1_scores_std = []    

    for i in range(len(samples)):

        completitude_means.append(np.mean(completitudes[i]))
        completitude_std.append(np.std(completitudes[i]))

        purity_means.append(np.mean(purities[i]))
        purity_std.append(np.std(purities[i]))

        f1_scores_means.append(np.mean(f1_scores[i]))
        f1_scores_std.append(np.std(f1_scores[i]))

    df = pd.DataFrame({
    'Galaxy_ID': df_sample_indexes,
    'completitud_mean': completitude_means,
    'completitud_std': completitude_std,
    'purity_mean': purity_means,
    'purity_std': purity_std,
    'f1_score_mean': f1_scores_means,
    'f1_score_std': f1_scores_std
    })

    return df