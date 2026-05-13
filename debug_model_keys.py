import torch
from utils.infer_utils import init_polynet_model, init_am_model
from problems import DVRPTW_Environment
import json
from argparse import Namespace

def main():
    path = "data/_PolyNet/chkpt_best.pyth"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    
    with open("data/_PolyNet/args.json") as f:
        args_dict = json.load(f)
    
    args = Namespace(**args_dict)
    
    if not hasattr(args, "customers_count"): args.customers_count = 100
    if not hasattr(args, "cust_k"): args.cust_k = 101
    if not hasattr(args, "greedy"): args.greedy = True
    
    p_learner = init_polynet_model(args, DVRPTW_Environment, "cpu")
    am_learner = init_am_model(args, DVRPTW_Environment, "cpu")
    
    p_keys = set(p_learner.state_dict().keys())
    am_keys = set(am_learner.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    
    print("Match with PolyNet:", len(p_keys & ckpt_keys))
    print("Match with AM:", len(am_keys & ckpt_keys))
    
    print("PolyNet missing:", len(p_keys - ckpt_keys))
    print("PolyNet unexpected:", len(ckpt_keys - p_keys))

if __name__ == "__main__":
    main()
