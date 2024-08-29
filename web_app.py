import torch
from dataset_webapp import MoleculeDataset
from model_webapp import ModelPredNumPeak, GINEGLOBAL
from add_external_global_features_webapp import add_morgan_fingerprint, add_daylight_fingerprint
from torch_geometric.loader import DataLoader
from utils_webapp import resume, count_parameters, enable_dropout, plot_spectrum
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
from data_postprocessing import post_precessing_data
from google_drive import get_model_from_drive


@st.cache_resource()
def load_model():
    drive_file_id_up = '1DgoulgBzQZE_FG-qYJ2Cn8Arznh2yTQo'
    drive_file_id_down = '1auEXPQo133aL1hSAArKQb7WC7vEcm0uQ'
    print('Loading model...')
    best_model_ckpt_up = get_model_from_drive(drive_file_id_up)
    print('Loading model...')
    best_model_ckpt_down = get_model_from_drive(drive_file_id_down)
    return best_model_ckpt_down, best_model_ckpt_up


def convert_df_to_csv(df):
    return df.to_csv(index=False, sep=',').encode('utf-8')


def plot_raman_spectrum(x, y, title='Raman Spectrum'):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, color='blue', label='Raman Spectrum')
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Raman Shift (cm$^{-1}$)', fontsize=12)
    ax.set_ylabel('Intensity (a.u.)', fontsize=12)
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)


def predict_num_of_peaks(smile, mc_sam=10):
    model_name = "ModelPredNumPeak"
    params = {}
    arch_params = {
        'dim_h': 256,
        'additional_feature_size': 12
    }
    n_data_points = 1
    dtf_predictions = pd.DataFrame({'smile': [smile]})

    for str_dataset in ['down', 'up']:
        dataset = MoleculeDataset(dtf_predictions)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)


        lst_file = [model for model in os.listdir(rf'web_app/pred_num_peak_{str_dataset}') if model.endswith('.pth')]
        lst_file.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
        best_model_ckpt = lst_file[-1]

        params["model_edge_dim"] = dataset[0].edge_attr.shape[1]

        # device = "cuda" if torch.cuda.is_available else "cpu"
        device = torch.device("cpu")

        model = eval(model_name)(
            node_feature_size=dataset[0].x.shape[1],
            edge_feature_size=params["model_edge_dim"],
            dim_h=arch_params['dim_h'],
            n_data_points=n_data_points,
            additional_feature_size=arch_params['additional_feature_size']
        )

        print("Number of params: ", count_parameters(model))
        model.to(device)

        resume(model, os.path.join(f'web_app/pred_num_peak_{str_dataset}', best_model_ckpt))
        model.eval()
        enable_dropout(model)

        for batch in tqdm(loader):
            lst_pred = []
            batch.to(device)
            # if len(batch.smiles) < 32: continue

            for i in range(mc_sam):
                pred = model(batch.x.float(),
                             None,
                             batch.edge_attr.float(),
                             batch.edge_index,
                             batch.batch)
                lst_pred.append(pred)

            pred = torch.mean(torch.stack(lst_pred, dim=2), dim=2)
            y_pred_batch = np.round(torch.squeeze(pred).cpu().detach().numpy())

        dtf_predictions[f'raman_pred_num_peak_{str_dataset}'] = y_pred_batch

    return dtf_predictions


def predict_raman_spectra(smile, model_down, model_up, mc_sam=10):
    # drive_file_id_up = '1DgoulgBzQZE_FG-qYJ2Cn8Arznh2yTQo'
    # drive_file_id_down = '1auEXPQo133aL1hSAArKQb7WC7vEcm0uQ'
    # model_down = torch.load(model_down, map_location=torch.device('cpu'))
    # model_up = torch.load(model_up, map_location=torch.device('cpu'))
    model_up.seek(0)
    model_down.seek(0)
    y_pred = []
    smiles = []
    num_peaks = []

    model_name = 'GINEGLOBAL'
    ext_feat_up = ["raman_pred_num_peak_up", "MOLECULAR_FINGERPRINT"]
    ext_feat_down = ["raman_pred_num_peak_down", "MOLECULAR_FINGERPRINT"]
    params = {
        "patience": 15,
        "batch_size": 1,
        "learning_rate": 0.01,
        "weight_decay": 0.001, "sgd_momentum": 0.9,
        "scheduler_gamma": 0.995,
        "model_embedding_size": 256,
        "model_attention_heads": 5,
        "model_layers": 4,
        "model_dropout_rate": 0.15,
        "model_top_k_ratio": 0.75,
        "model_top_k_every_n": 1,
        "model_dense_neurons": 128}
    arch_params= {
        "dim_h": 256,
        "additional_feature_size": 2049}

    # for str_dataset in ['down', 'up']:
    for str_dataset, best_model_ckpt in zip(['down', 'up'], [model_down, model_up]):
        y_pred = []

        if str_dataset == 'down':
            test_dataset = MoleculeDataset(smile, additional_feat=ext_feat_down)
            # best_model_ckpt = get_model_from_drive(drive_file_id_down)
        else:
            test_dataset = MoleculeDataset(smile, additional_feat=ext_feat_up)
            # best_model_ckpt = get_model_from_drive(drive_file_id_up)


        test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=False)

        n_data_points = 267

        # # Load model from directory path
        # lst_file = [model for model in os.listdir(f'web_app/spectra_predictions_{str_dataset}') if model.endswith('.pth')]
        # lst_file.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
        # best_model_ckpt = lst_file[-1]

        params["model_edge_dim"] = test_dataset[0].edge_attr.shape[1]

        device = torch.device("cpu")
        model = eval(model_name)(node_feature_size=test_dataset[0].x.shape[1],
                                 edge_feature_size=test_dataset[0].edge_attr.shape[1],
                                 n_data_points=n_data_points, **arch_params)
        print("Number of params: ", count_parameters(model))
        model.to(device)

        # # Load model from directory path
        # resume(model, os.path.join(f'web_app/spectra_predictions_{str_dataset}', best_model_ckpt))

        # Load from Drive the model, for every prediction
        model.load_state_dict(torch.load(best_model_ckpt, map_location=torch.device('cpu')))
        model.eval()
        enable_dropout(model)

        for batch in tqdm(test_loader):
            lst_pred = []
            batch.to(device)

            for i in range(mc_sam):
                pred = model(batch.x.float(),
                             batch.graph_level_feats,
                             batch.edge_attr.float(),
                             batch.edge_index,
                             batch.batch)
                lst_pred.append(pred)

            pred = torch.mean(torch.stack(lst_pred, dim=2), dim=2)
            y_pred_batch = torch.squeeze(pred).cpu().detach().numpy()

            y_pred.extend(y_pred_batch)
            num_peaks.extend(batch.graph_level_feats.reshape(len(batch.smiles), -1)[:, 0].cpu().detach().numpy().tolist())
            smiles.extend(batch.smiles)

        smile[f'raman_pred_{str_dataset}'] = [y_pred]
    return smile


def is_valid_smile(smile: str) -> bool:
    mol = Chem.MolFromSmiles(smile)
    return mol is not None


model_down, model_up = load_model()
st.title('Prediction of number of  raman peaks starting from a SMILE')
st.write('Enter the SMILE representation of a molecule')

smile = st.text_area('Insert SMILE')

# Prediction and display result
if st.button('Predict'):
    if not is_valid_smile(smile):
        st.error("Invalid SMILES string. Please enter a valid SMILES representation.")
    else:
        molecule = Chem.MolFromSmiles(smile)
        img = Draw.MolToImage(molecule)
        # Display the image in Streamlit
        # st.image(img, caption="Molecule from SMILES", width=300)
        col1, col2, col3 = st.columns([1, 2, 1])

        # Center the image in the middle column
        with col2:
            st.image(img, caption="Molecule from SMILES", use_column_width=True)

        dtf_number_of_peaks = predict_num_of_peaks(smile)
        dtf_prediction = add_morgan_fingerprint(dtf_number_of_peaks)
        dtf_prediction = add_daylight_fingerprint(dtf_prediction)
        dtf_raman_spectra = predict_raman_spectra(dtf_prediction, model_down, model_up)

        dtf_prediction = post_precessing_data(dtf_raman_spectra)

        # Convert DataFrame to CSV
        csv_data = convert_df_to_csv(dtf_prediction)

        # Plot the Raman Spectra
        raman_spectra = dtf_prediction['raman_pred'].iloc[0]
        len_sp = len(dtf_prediction['raman_pred'].iloc[0])
        x = np.linspace(500, 3500, len_sp)
        plot_spectrum(raman_spectra, 500, 3500, rescale=3)

        # Download button
        st.download_button(
            label="Download data as CSV",
            data=csv_data,
            file_name='raman_spectrum.csv',
            mime='text/csv',
        )

        # st.write(f'Predicted fingerprint region peaks: {dtf_number_of_peaks.raman_pred_num_peak_down.iloc[0]}')
        # st.write(f'Predicted CH region peaks: {dtf_number_of_peaks.raman_pred_num_peak_up.iloc[0]}')
