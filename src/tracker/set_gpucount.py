from typing import Dict, Any, Tuple, Optional
from easydict import EasyDict
import json
import re

# 定数
TEAM_CONFIGS = {
    "abeja-geniac": (("NUM_NODES", "trainer.num_nodes"), None),
    "aidealab-geniac": ("gpus", None),
    "aihub-geniac": ("nnodes", None),
    "aiinside-geniac": ("nnodes", None),
    "alt-geniac": ("NNODES", None),
    "datagrid-geniac": None,
    "eques-geniac": None,
    "future-geniac": ("world_size", None),
    "humanome-geniac": None,
    "jamstec-geniac": None,
    "karakuri-geniac": ("world_size", None),
    "kotoba-geniac": ("num_nodes", "num_gpus"),
    "nablas-geniac": ("num_nodes", "num_gpus_per_node"),
    "ricoh-geniac": ("NNODES", "NUM_GPUS"),
    "stockmark-geniac": None,
    "syntheticgestalt-geniac": None,
    "ubitus-geniac": None
}

def get_config_value(config: Dict[str, Any], key: str) -> int:
    """設定から値を取得し、整数に変換する"""
    value = config.get(key, {}).get('value', 0)
    try:
        return int(value)
    except (ValueError, TypeError):
        print(f"Could not convert '{value}' to int. Using 0 instead.")
        return 0

def get_config_value_multi(config: Dict[str, Any], keys: Tuple[str, ...]) -> int:
    """複数のキーから最初に見つかった値を取得し、整数に変換する"""
    for key in keys:
        value = get_config_value(config, key)
        if value != 0:
            return value
    return 0

def calculate_gpu_count(num_nodes: int, gpu_key: Optional[str], config_dict: Dict[str, Any], team: str, node: EasyDict) -> int:
    """GPUカウントを計算する"""
    if team in ["abeja-geniac", "alt-geniac", "aiinside-geniac"]:
        return num_nodes * 8
    elif team == "ricoh-geniac":
        run_name = node.description
        match = re.match(r'(\d+)Node', run_name)
        if match:
            num_nodes = int(match.group(1))
        else:
            num_nodes = 1
        return num_nodes * 8
    elif team == "aidealab-geniac":
        summary_dict = json.loads(node.summaryMetrics)
        gpu_count = summary_dict.get("gpus", 0)
        return gpu_count
    elif team =="aihub-geniac":
        num_nodes_str = node.description if node else "0Node"
        num_gpus = node.runInfo.gpuCount if node.runInfo else 0
        if num_nodes_str and num_nodes_str[-5].isdigit() and num_nodes_str.endswith("Node"):
            num_nodes = int(num_nodes_str[-5])
        else:
            num_nodes = 0
        return num_nodes * num_gpus
    elif gpu_key:
        num_gpus = get_config_value(config_dict, gpu_key)
        if num_gpus == 0:
            print(f"num_gpus is 0 for {team} ({node.name}). Using num_nodes as GPU count.")
            return num_nodes
        return num_nodes * num_gpus
    else:
        return num_nodes

def set_gpucount(node: EasyDict, team: str) -> int:
    """チームごとのGPUカウントを設定する"""
    default_gpu_count = node.runInfo.gpuCount if node.runInfo else 0
    
    if team not in TEAM_CONFIGS:
        print(f"Unknown team {team}. Using default GPU count.")
        return default_gpu_count

    config_dict = json.loads(node.config)
    
    try:
        team_config = TEAM_CONFIGS[team]
        if team_config is None:
            return default_gpu_count
        
        node_key, gpu_key = team_config
        num_nodes = get_config_value_multi(config_dict, node_key) if isinstance(node_key, tuple) else get_config_value(config_dict, node_key)
        
        if num_nodes == 0:
            print(f"num_nodes is 0 for {team} ({node.name}).")
        
        gpu_count = calculate_gpu_count(num_nodes, gpu_key, config_dict, team, node)
        
        print(f"Calculated GPU count for {team} ({node.name}): {gpu_count}")
        return gpu_count if gpu_count > 0 else default_gpu_count
    
    except Exception as e:
        print(f"Error calculating GPU count for {team} ({node.name}): {str(e)}")
        return default_gpu_count