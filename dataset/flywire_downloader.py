"""
MaleCNS v1.0 & FlyWire CAVE API Connector & Dynamic Query System
Avoids the 'Root ID Trap' by querying through Supervoxel IDs and Cell-Type Annotations
"""
import os
import sys
import json
import requests
from typing import Optional, Dict, Any, List

class FlyWireDownloader:
    """
    Secure access and retrieval utility for FlyWire & MaleCNS v1.0 databases.
    
    IMPORTANT (Root ID Trap Resolution):
    In FlyWire and MaleCNS databases, `root_id` identifiers frequently change as neurons
    are split or merged during proofreading iterations.
    Hardcoding root_id lists from specific releases (e.g. v630) causes broken queries.
    
    This class dynamically queries the CAVE API using invariant `supervoxel_id`
    or canonical cell-type annotations (e.g. 'DNp20', 'DNpe017') to retrieve
    the latest active root IDs.
    """
    def __init__(self, target_dir: Optional[str] = None):
        self.target_dir = target_dir or os.path.join(os.path.dirname(__file__), "..", "cache")
        os.makedirs(self.target_dir, exist_ok=True)
        self.datastack_name = "flywire_fafb_production"

        # Canonical Cell Types
        self.canonical_cell_types = {
            "DNp20": ["DNp20_R", "DNp20_L"],
            "DNpe017": ["DNpe017_R", "DNpe017_L"],
            "PPL1": ["PPL101", "PPL102", "PPL103"],
            "KC": ["KCg", "KCa'b'", "KCab"]
        }

    def check_caveclient(self) -> bool:
        """Checks if caveclient library is installed."""
        try:
            import caveclient
            return True
        except ImportError:
            return False

    def query_latest_root_ids(self, supervoxel_ids: List[int], token: Optional[str] = None) -> List[int]:
        """
        Uses CAVE ChunkedGraph API to map invariant supervoxel IDs
        to the current dataset root IDs (Root ID Trap Resolution).
        """
        if not self.check_caveclient():
            print("[Dataset] caveclient library not found. Install via: pip install caveclient")
            return []

        import caveclient
        try:
            client = caveclient.CAVEclient(self.datastack_name, auth_token=token)
            current_roots = client.chunkedgraph.get_roots(supervoxel_ids)
            print(f"[Dataset] Retrieved latest root_ids for {len(supervoxel_ids)} supervoxels.")
            return list(set(current_roots))
        except Exception as e:
            print(f"[Dataset] CAVE get_roots query error: {e}")
            return []

    def query_by_cell_type(self, cell_type: str = "DNp20", token: Optional[str] = None) -> Optional[List[int]]:
        """
        Queries CAVE annotation tables by canonical cell type (e.g. DNp20)
        to retrieve the latest verified root IDs.
        """
        if not self.check_caveclient():
            return None

        import caveclient
        try:
            client = caveclient.CAVEclient(self.datastack_name, auth_token=token)
            ann_df = client.materialize.query_table(
                "hierarchical_proofreading_v1",
                filter_equal_dict={"cell_type": cell_type}
            )
            if ann_df is not None and not ann_df.empty:
                root_ids = ann_df["pt_root_id"].tolist()
                print(f"[Dataset] Found {len(root_ids)} active root_ids for type '{cell_type}'.")
                return root_ids
        except Exception as e:
            print(f"[Dataset] Annotation query error: {e}")
            return None

    def print_data_guide(self):
        guide = """
========================================================================
     MaleCNS v1.0 & FlyWire Dynamic Neuron Mapping Guide
========================================================================
[!] ROOT ID TRAP WARNING:
    In FlyWire, 18-digit root_id values evolve across dataset releases.
    Avoid copying hardcoded static root_id lists. Instead:
    
    1. Query By Cell-Type Annotation:
       client.materialize.query_table('hierarchical_proofreading_v1', 
                                      filter_equal_dict={'cell_type': 'DNp20'})
                                      
    2. Map Invariant Supervoxel IDs to Roots:
       current_root = client.chunkedgraph.get_roots(supervoxel_id)

    3. This simulation engine natively compiles the 139,000+ neuron
       connectome into SciPy CSR and PyTorch Sparse Tensor formats.
========================================================================
"""
        print(guide)

if __name__ == "__main__":
    downloader = FlyWireDownloader()
    downloader.print_data_guide()
