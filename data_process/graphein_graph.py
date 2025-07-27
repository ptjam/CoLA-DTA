from functools import partial
from graphein.protein.graphs import construct_graph
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.edges.distance import add_distance_threshold
from torch_geometric.data import Data
from torch_geometric.transforms import AddLaplacianEigenvectorPE
from pathlib import Path
from Bio.PDB import PDBParser, DSSP
from typing import Dict, List, Tuple
import logging
import warnings
from typing import Union
import torch
from tqdm import tqdm
import concurrent.futures



class ProteinGraphConverter:
    def __init__(self, dssp_path="/home/cuichen/anaconda3/envs/dta/bin/mkdssp", lap_dim=8):
        # 初始化日志和警告设置
        logging.getLogger("graphein").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore")
        warnings.simplefilter(action="ignore", category=FutureWarning)

        # 氨基酸特征映射
        self.aa_list = [
            "ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU",
            "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR", "UNK"
        ]
        self.aa2onehot = {aa: [int(i == idx) for i in range(21)] for idx, aa in enumerate(self.aa_list)}

        self.polarity_map = {
            "ARG": [1, 0, 0], "LYS": [1, 0, 0], "HIS": [1, 0, 0],
            "ASP": [0, 1, 0], "GLU": [0, 1, 0]
        }

        self.hydrophobicity = {
            "ALA": 1.8, "CYS": 2.5, "ASP": -3.5, "GLU": -3.5, "PHE": 2.8,
            "GLY": -0.4, "HIS": -3.2, "ILE": 4.5, "LYS": -3.9, "LEU": 3.8,
            "MET": 1.9, "ASN": -3.5, "PRO": -1.6, "GLN": -3.5, "ARG": -4.5,
            "SER": -0.8, "THR": -0.7, "VAL": 4.2, "TRP": -0.9, "TYR": -1.3
        }

        # Graphein 配置
        edge_func = partial(add_distance_threshold, threshold=5.5, long_interaction_threshold=5.5)
        self.config = ProteinGraphConfig(
            granularity="centroids",
            insertions=False,
            edge_construction_functions=[edge_func]
        )
        self.dssp_path = dssp_path
        self.lap_dim = lap_dim  # 控制拉普拉斯特征维度

        # 实例化 Laplacian PE transform
        self.lap_transform = AddLaplacianEigenvectorPE(k=self.lap_dim, attr_name="lap_pos_enc", is_undirected=True)

    def extract_dssp_features(self, pdb_path: str, dssp_exe=None) -> Dict[Tuple[str, int], List[float]]:
        dssp_exe = dssp_exe or self.dssp_path
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_path)
        model = structure[0]
        dssp = DSSP(model, pdb_path, dssp=dssp_exe)

        dssp_dict = {}
        for key in dssp.keys():
            chain_id, res_id = key
            dssp_data = dssp[key]

            ss_raw = dssp_data[2] or "C"
            ss = ss_raw[0] if ss_raw in "HE" else "C"
            ss_onehot = {"H": [1, 0, 0], "E": [0, 1, 0], "C": [0, 0, 1]}.get(ss, [0, 0, 1])

            phi = max(min(float(dssp_data[4]), 180.0), -180.0) if dssp_data[4] else 0.0
            psi = max(min(float(dssp_data[5]), 180.0), -180.0) if dssp_data[5] else 0.0

            phi_norm = round(phi / 180.0, 2)
            psi_norm = round(psi / 180.0, 2)

            dssp_dict[(chain_id, res_id[1])] = ss_onehot + [phi_norm, psi_norm]

        return dssp_dict # 5

    def get_residue_features(self, res, dssp_dict=None):
        resname = res.get("residue_name", "UNK")
        resname = resname if resname in self.aa2onehot else "UNK"
        one_hot = self.aa2onehot[resname]

        polarity = self.polarity_map.get(resname, [0, 0, 1])
        hydro = [round(self.hydrophobicity.get(resname, 0.0), 2)]

        chain = res.get("chain_id", "A")
        res_idx = res.get("residue_number", 0)
        dssp_feat = dssp_dict.get((chain, res_idx), [0, 0, 1, 0.0, 0.0])

        return torch.tensor(one_hot + polarity + hydro + dssp_feat, dtype=torch.float) # 21+3+1+5

    def pdb_to_pyg_graph(self, pdb_path: str, dssp_exe=None) -> Data:
        dssp_exe = dssp_exe or self.dssp_path
        dssp_dict = self.extract_dssp_features(pdb_path, dssp_exe)
        g = construct_graph(config=self.config, pdb_path=pdb_path)

        node_list = list(g.nodes())
        node_idx_map = {node: idx for idx, node in enumerate(node_list)}
        x = torch.stack([self.get_residue_features(g.nodes[node], dssp_dict) for node in node_list])

        edge_index = []
        for src, dst in g.edges():
            i, j = node_idx_map[src], node_idx_map[dst]
            edge_index.append([i, j])
        edge_index = torch.tensor(edge_index, dtype=torch.long).T

        data = Data(x=x, edge_index=edge_index)

        # 使用官方 AddLaplacianEigenvectorPE 添加拉普拉斯特征
        data = self.lap_transform(data)

        # 将拉普拉斯特征拼接到原始节点特征上
        if hasattr(data, 'lap_pos_enc'):
            data.x = torch.cat([data.x, data.lap_pos_enc], dim=1)
            del data.lap_pos_enc
        else:
            raise ValueError("Laplacian positional encoding not found in data after applying transform.")

        return data

class ProteinGraphBatchSaver:
    def __init__(self, converter_cls, pdb_root: Union[str, Path], save_root: Union[str, Path], max_workers: int = 8):
        """
        :param converter_cls: ProteinGraphConverter 类（非实例，便于多进程安全构造）
        :param pdb_root: 包含多个蛋白质子文件夹
        :param save_root:
        :param max_workers: 并发进程数
        """
        self.converter_cls = converter_cls
        self.pdb_root = Path(pdb_root)
        self.save_protein = Path(save_root)  # 不再追加 /protein
        self.save_protein.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers

    def _process_single_file(self, pdb_file: Path):
        if not pdb_file.name.endswith(".pdb"):
            return f"[Skip] {pdb_file.name} is not a .pdb file"

        pdb_id = pdb_file.stem.upper()
        output_path = self.save_protein / f"{pdb_id}.pt"

        if output_path.exists():
            return f"[Skip] Already processed {pdb_id}"

        try:
            converter = self.converter_cls()
            protein_graph = converter.pdb_to_pyg_graph(str(pdb_file))
        except Exception as e:
            return f"[Error] Failed {pdb_id}: {e}"

        torch.save(protein_graph, output_path)
        return f"[OK] Processed {pdb_id}"

    def process_all(self):
        pdb_files = sorted(self.pdb_root.glob("*.pdb"))
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(tqdm(executor.map(self._process_single_file, pdb_files), total=len(pdb_files)))

        for res in results:
            print(res)


if __name__ == "__main__":
    converter = ProteinGraphConverter()
    saver = ProteinGraphBatchSaver(
        converter_cls=ProteinGraphConverter,  # 注意不要加 ()
        pdb_root="../data/KIBA/kiba_pdb_files",
        save_root="../data/KIBA/protein_graph"
    )

    saver.process_all()
