import os
import pandas as pd
import numpy as np
import torch
import h5py
from tqdm import tqdm
from pathlib import Path
from rdkit import Chem
from transformers import AutoTokenizer, RobertaModel
from torch_geometric.data import Data

class SmilesToGraphEmbedding:
    def __init__(self, model_name="DeepChem/ChemBERTa-77M-MLM", device=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.atom_types = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
                           'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co',
                           'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                           'Pt', 'Hg', 'Pb', 'Unknown']
        self.degree_set = list(range(11))
        self.hydrogen_set = list(range(11))
        self.valence_set = list(range(11))

    def one_of_k_encoding(self, x, allowable_set):
        if x not in allowable_set:
            raise ValueError(f"{x} not in {allowable_set}")
        return [x == s for s in allowable_set]

    def one_of_k_encoding_unk(self, x, allowable_set):
        if x not in allowable_set:
            x = allowable_set[-1]
        return [x == s for s in allowable_set]

    def atom_features(self, atom):
        return self.one_of_k_encoding_unk(atom.GetSymbol(), self.atom_types) + \
               self.one_of_k_encoding(atom.GetDegree(), self.degree_set) + \
               self.one_of_k_encoding_unk(atom.GetTotalNumHs(), self.hydrogen_set) + \
               self.one_of_k_encoding_unk(atom.GetImplicitValence(), self.valence_set) + \
               [atom.GetIsAromatic()]

    def bond_features(self, bond):
        bt = bond.GetBondType().name
        return [
            int(bt == 'SINGLE'),
            int(bt == 'DOUBLE'),
            int(bt == 'TRIPLE'),
            int(bond.GetIsAromatic()),
            int(bond.IsInRing()),
            int(bond.GetIsConjugated())
        ]

    def smiles_to_graph(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        node_feats = [self.atom_features(atom) for atom in mol.GetAtoms()]
        x = torch.tensor(np.array(node_feats), dtype=torch.float)

        edge_index = []
        edge_attr = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            feat = self.bond_features(bond)
            edge_index += [[i, j], [j, i]]
            edge_attr += [feat, feat]

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def get_embedding(self, smiles):
        with torch.no_grad():
            inputs = self.tokenizer(smiles, return_tensors="pt", truncation=True, padding=True).to(self.device)
            outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()


def process_tsv(tsv_path, output_graph_dir, output_h5_path):
    os.makedirs(output_graph_dir, exist_ok=True)
    df = pd.read_csv(tsv_path, sep="\t")

    # 确保 compound_id 为字符串
    df["compound_id"] = df["compound_id"].astype(str)

    # 去重：cid → SMILES 映射（避免重复处理相同 cid）
    cid_to_smiles = dict(df[["compound_id", "SMILES"]].drop_duplicates().values)

    graph_builder = SmilesToGraphEmbedding()

    if os.path.exists(output_h5_path):
        print(f"⚠️ Removing old file: {output_h5_path}")
        os.remove(output_h5_path)

    with h5py.File(output_h5_path, 'w') as h5_file:
        for cid, smiles in tqdm(cid_to_smiles.items(), desc="Processing unique compounds"):
            graph_path = os.path.join(output_graph_dir, f"{cid}.pt")
            try:
                # 保存图
                if not os.path.exists(graph_path):
                    graph = graph_builder.smiles_to_graph(smiles)
                    torch.save(graph, graph_path)
                else:
                    print(f"[Skipped Graph] {cid} already exists")

                # 保存嵌入（键必须为 str）
                if cid not in h5_file:
                    emb = graph_builder.get_embedding(smiles)
                    h5_file.create_dataset(cid, data=emb)
                else:
                    print(f"[Skipped H5] {cid} already in h5 file")

            except Exception as e:
                print(f"[Error] {cid}: {e}")

    print("✅ All compounds processed.")



if __name__ == "__main__":
    process_tsv(
        tsv_path="../data/KIBA/kiba_with_smiles_and_seq.tsv",
        output_graph_dir="../data/KIBA/compound_graphs",
        output_h5_path="../data/KIBA/compound_bert_features.h5"
    )
