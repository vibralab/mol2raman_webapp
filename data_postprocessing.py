import pandas as pd
import numpy as np
from utils_webapp import generate_lorentzian_kernel, make_conv_matrix, sim, rescale, f1_score_mod, \
    convolve_with_lorentzian, keep_peaks_prom, lorentz_conv, spectral_information_similarity, \
    count_matched_peaks, sim_dir, precision_score_mod, recall_score_mod,  calc_cos_sim, fnr_score_mod, \
    convolve_with_lorentzian_padded
from sklearn.metrics import root_mean_squared_error, r2_score, mean_squared_error
import pickle



def process_arrays(df, col1, col2, new_column):
    def create_combined_array(arr1, arr2):
        first_700 = arr1[:700]
        last_700 = arr2[-700:]

        last_200_of_450 = arr1[-100:]
        first_200_of_900 = arr2[:100]

        mean_200 = (last_200_of_450 + first_200_of_900) / 2

        combined_array = np.concatenate([first_700, mean_200, last_700])

        return combined_array

    df[new_column] = [create_combined_array(arr1, arr2) for arr1, arr2 in zip(df[col1], df[col2])]

    return df


def post_precessing_data(df):
    # Rescale raman_pred and raman_true
    df['raman_pred_up'] = df['raman_pred_up'].apply(lambda row: rescale(range(0, 800), row))
    df['raman_pred_down'] = df['raman_pred_down'].apply(lambda row: rescale(range(0, 800), row))


    # # Rescale Intensities (only if the intensities had been rescaled in the split
    # df_down['raman_pred'] = df_down.apply(lambda row: row['raman_pred']/5, axis=1)
    # df_down['raman_true'] = df_down.apply(lambda row: row['raman_true']/5, axis=1)

    # Apply keep_peaks_prom
    df['raman_pred_down'] = df.apply(lambda row: keep_peaks_prom(row.raman_pred_down, round(row.raman_pred_num_peak_down)), axis=1)
    df['raman_pred_up'] = df.apply(lambda row: keep_peaks_prom(row.raman_pred_up, round(row.raman_pred_num_peak_up)), axis=1)

    # Merge into a unique Dataframe for all Spectrum, first 700 down + 200 mean up&down + last 700
    result_df = process_arrays(df, 'raman_pred_down', 'raman_pred_up', 'raman_pred')

    gamma = 2.5
    kernel_size = 600
    lorentzian_kernel = generate_lorentzian_kernel(kernel_size, gamma)
    leng = list(range(501, 3500, 2))
    conv = make_conv_matrix(std_dev=10, frequencies=leng)

    result_df['raman_pred_conv'] = result_df.apply(
        lambda row: convolve_with_lorentzian(row['raman_pred'], lorentzian_kernel), axis=1)

    return result_df
