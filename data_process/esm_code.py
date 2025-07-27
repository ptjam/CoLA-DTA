import os
import logging
import warnings

import esm
import h5py
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from rdkit import Chem
from Bio.PDB import PDBParser, PPBuilder

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("esm_processing.log")
    ]
)
logger = logging.getLogger(__name__)


class PDBToESMProcessor:
    def __init__(self, pdb_folder, output_fasta, output_hdf5, processed_log="processed_ids.txt"):
        """
        :param pdb_folder: 包含 PDB 文件的文件夹路径
        :param output_fasta: 输出的 FASTA 文件路径
        :param output_hdf5: 输出的 HDF5 文件路径
        :param processed_log: 已处理蛋白 ID 的记录文件
        """
        self.pdb_folder = Path(pdb_folder)
        self.output_fasta = Path(output_fasta)
        self.output_hdf5 = Path(output_hdf5)
        self.processed_log = Path(processed_log)

        # 初始化模型和字典
        self.model_name = "esm2_t33_650M_UR50D"
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet_hub(self.model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 确保输出目录存在
        self.output_hdf5.parent.mkdir(parents=True, exist_ok=True)
        self.output_fasta.parent.mkdir(parents=True, exist_ok=True)

    def load_processed_ids(self):
        """从日志中加载已处理的蛋白ID"""
        if not self.processed_log.exists():
            return set()
        with open(self.processed_log, "r") as f:
            return set(line.strip() for line in f.readlines())

    def save_processed_id(self, protein_id):
        """将蛋白ID写入已处理日志"""
        with open(self.processed_log, "a") as f:
            f.write(f"{protein_id}\n")

    def pdb_to_sequence(self, pdb_path):
        """使用 RDKit 和 Biopython 提取氨基酸序列"""
        try:
            # Step 1: 使用 RDKit 获取残基名（可能包含非标准残基）
            mol = Chem.MolFromPDBFile(str(pdb_path), sanitize=False, removeHs=True)
            resnames = [atom.GetPDBResidueInfo().GetResidueName().strip() for atom in mol.GetAtoms() if
                        atom.GetPDBResidueInfo()]

            # Step 2: 使用 Biopython 获取标准氨基酸序列
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('protein', str(pdb_path))
            ppb = PPBuilder()
            seq = ''.join([str(pp.get_sequence()) for pp in ppb.build_peptides(structure)])

            if not seq:
                raise ValueError("No standard sequence found using Biopython")

            return seq

        except Exception as e:
            logger.warning(f"[Fallback] Using residue names from RDKit: {str(e)}")
            # Fallback: 直接返回唯一残基名拼接（不推荐，仅调试）
            unique_res = list(dict.fromkeys(resnames))
            return ''.join([self._three_to_one(r) for r in unique_res if r in self._aa_map])

    _aa_map = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }

    def _three_to_one(self, resname):
        return self._aa_map.get(resname.upper(), 'X')  # X 表示未知残基

    def batch_process_pdbs(self):
        """批量提取 PDB 中的氨基酸序列并保存为 FASTA 文件"""
        logger.info("🔄 Step 1: Converting PDB files to FASTA format")

        pdb_files = list(self.pdb_folder.rglob("*.pdb"))
        total = len(pdb_files)
        logger.info(f"Found {total} PDB files")

        processed_ids = self.load_processed_ids()
        remaining_files = [f for f in pdb_files if f.stem not in processed_ids]
        logger.info(f"{len(processed_ids)} already processed, {len(remaining_files)} left")

        with open(self.output_fasta, "w") as fasta_file:
            for pdb_file in tqdm(remaining_files, desc="Extracting sequences", unit="file"):
                try:
                    protein_id = pdb_file.stem
                    seq = self.pdb_to_sequence(pdb_file)
                    fasta_file.write(f">{protein_id}\n{seq}\n")
                    self.save_processed_id(protein_id)
                    logger.info(f"[Processed] {protein_id} ({len(seq)} AA)")
                except Exception as e:
                    logger.error(f"[Failed] {pdb_file.name}: {str(e)}")

        logger.info("✅ PDB to FASTA conversion completed")

    def run_esm_embedding(self, sequences, per_residue=True):
        """使用 ESM 模型生成嵌入"""
        results = {}

        batch_converter = self.alphabet.get_batch_converter()
        self.model.to(self.device).eval()

        batch_labels, batch_strs, batch_tokens = batch_converter(list(sequences.items()))
        batch_tokens = batch_tokens.to(self.device)

        with torch.no_grad():
            outputs = self.model(batch_tokens, repr_layers=[33])
            embeddings = outputs["representations"][33]

        for i, protein_id in enumerate(batch_labels):
            if per_residue:
                emb = embeddings[i, 1:-1].cpu().numpy()  # 去掉 <cls> 和 <eos>
            else:
                emb = embeddings[i].mean(dim=0).cpu().numpy()  # 全图平均向量
            results[protein_id] = emb

        return results

    def save_embeddings_to_hdf5(self, embeddings_dict):
        """将嵌入保存为 HDF5 格式"""
        with h5py.File(self.output_hdf5, "a") as f:
            for prot_id, emb in embeddings_dict.items():
                if prot_id in f:
                    del f[prot_id]
                f.create_dataset(prot_id, data=emb)
        logger.info(f"Embeddings saved to {self.output_hdf5}")

    def batch_process_with_esm(self, batch_size=4, per_residue=True):
        """批量处理 FASTA 文件中的蛋白序列，并使用 ESM 模型提取嵌入"""
        logger.info("🔄 Step 2: Generating ESM embeddings from FASTA file")

        # 加载序列
        sequences = {}
        with open(self.output_fasta, "r") as f:
            lines = [line.strip() for line in f.readlines()]
        i = 0
        while i < len(lines):
            if lines[i].startswith(">"):
                protein_id = lines[i][1:]
                i += 1
                seq = ""
                while i < len(lines) and not lines[i].startswith(">"):
                    seq += lines[i]
                    i += 1
                sequences[protein_id] = seq
            else:
                i += 1
        logger.info(f"Loaded {len(sequences)} sequences from {self.output_fasta}")

        # 分批处理
        protein_ids = list(sequences.keys())
        success_count = 0

        with h5py.File(self.output_hdf5, "a") as f:
            pbar = tqdm(range(0, len(protein_ids), batch_size), desc="Processing with ESM")
            for i in pbar:
                batch_ids = protein_ids[i:i + batch_size]
                batch_seqs = {pid: sequences[pid] for pid in batch_ids}
                try:
                    embeddings = self.run_esm_embedding(batch_seqs, per_residue=per_residue)
                    for pid, emb in embeddings.items():
                        if pid in f:
                            del f[pid]
                        f.create_dataset(pid, data=emb)
                        self.save_processed_id(pid)
                        success_count += 1
                        pbar.set_postfix({"Current": pid})
                    logger.info(f"[Success] Batch processed: {', '.join(batch_ids)}")
                except Exception as e:
                    logger.error(f"[Failed] Batch failed: {str(e)}")

        logger.info(f"✅ ESM feature extraction complete: {success_count}/{len(sequences)} succeeded")


if __name__ == "__main__":
    processor = PDBToESMProcessor(
        pdb_folder="../data/DAVIS/davis_pdb_files",
        output_fasta="../data/DAVIS/davis_protein_sequences.fasta",
        output_hdf5="../data/DAVIS/davis_esm_embeddings.h5",
        processed_log="../data/DAVIS/davis_processed_ids.txt"
    )

    # Step 1: PDB → FASTA
    # processor.batch_process_pdbs()

    # Step 2: FASTA → ESM Embedding → HDF5
    processor.batch_process_with_esm(batch_size=1, per_residue=False)