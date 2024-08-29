import pandas as pd
import torch
from rdkit import Chem
from utils_webapp import featurize_with_retry
import deepchem as dc
import torch_geometric
import numpy as np
from torch_geometric.data import Dataset
from custom_featurizer import custom_featuriz

print(f"Torch version: {torch.__version__}")
# print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")

class MoleculeDataset(Dataset):
    def __init__(self, smiles, featuriz: str = 'MolGraphConv', additional_feat: list = None):
        self.smile = smiles
        self.featuriz = featuriz
        self.additional_features = additional_feat
        self.data = [self.process()]
        # self.data = [self.featurize_single_molecule(smiles.smile.iloc[0])]  # Wrap single molecule in a list
        super(MoleculeDataset, self).__init__(root='.', transform=None, pre_transform=None)

    @staticmethod
    def molecule_from_smiles(smiles):
        molecule = Chem.MolFromSmiles(smiles.smile.iloc[0], sanitize=False)
        flag = Chem.SanitizeMol(molecule, catchErrors=True)
        if flag != Chem.SanitizeFlags.SANITIZE_NONE:
            Chem.SanitizeMol(molecule, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ flag)
        Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
        return molecule

    # @property
    # def raw_file_names(self):
    #     return []  # No raw files
    #
    # @property
    # def processed_file_names(self):
    #     return []

    def process(self):
        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True, use_chirality=True, use_partial_charge=True)
        mol = MoleculeDataset.molecule_from_smiles(self.smile)
        try:
            f = featurize_with_retry(featurizer, mol)
        except Exception as e:
            print(f'Error featurizing smile: {self.smile}')
            print(mol)
            raise e

        data = f.to_pyg_graph()
        data.smiles = self.smile

        selected_cols = self.additional_features
        selected_values = []

        if self.additional_features is not None:
            for col in self.additional_features:
                val = self.smile[col].values

                if isinstance(val[0], list):
                    selected_values.extend(val[0])
                else:
                    selected_values.append(val[0])

            selected_values = np.array(selected_values, dtype=np.float64)
            torch_tensor = torch.from_numpy(selected_values)
            data.graph_level_feats = torch_tensor

        return data

    def len(self):
        return 1  # Returns 1, since it's handling a single molecule

    def get(self, idx):
        return self.data[0]

    @property
    def raw_file_names(self):
        # Return a dummy list, as you don't need raw files
        return []

    @property
    def processed_file_names(self):
        # Return a dummy list, as you're not processing files
        return []

    def download(self):
        # Override this to do nothing, as you don't need to download data
        pass
