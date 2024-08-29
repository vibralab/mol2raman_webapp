import json
import pandas as pd
import numpy
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator


def add_external_feature(dtf_in, dtf_feat, str_feat, str_rename):

    dtf_out = dtf_in.copy()
    dtf_out = dtf_out.merge(dtf_feat[['SMILE', str_feat]], on='SMILE', how='inner')
    dtf_out.rename({str_feat: str_rename}, axis=1, inplace=True)

    return dtf_out


def add_daylight_fingerprint(dtf_in):

    dtf_out = dtf_in.copy()
    lst_smiles = dtf_out.smile.tolist()

    fpgen = AllChem.GetRDKitFPGenerator(fpSize=2048)
    dct_fingerprint = {k: [list(fpgen.GetFingerprint(Chem.MolFromSmiles(k)))]
                       for k in lst_smiles}

    dtf_tmp = pd.DataFrame.from_dict(dct_fingerprint, orient='index', columns=['MOLECULAR_FINGERPRINT'])
    dtf_out = dtf_out.merge(dtf_tmp, how='left', left_on='smile', right_index=True)

    return dtf_out


def add_morgan_fingerprint(dtf_in):

    dtf_out = dtf_in.copy()
    lst_smiles = dtf_out.smile.tolist()

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=3)
    dct_fingerprint = {k: [list(gen.GetFingerprint(Chem.MolFromSmiles(k)))]
                       for k in lst_smiles}

    dtf_tmp = pd.DataFrame.from_dict(dct_fingerprint, orient='index', columns=['MOLECULAR_FINGERPRINT_MORGAN'])
    dtf_out = dtf_out.merge(dtf_tmp, how='left', left_on='smile', right_index=True)

    return dtf_out