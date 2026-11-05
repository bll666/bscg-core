"""Write the current runtime metadata, without any dataset content."""
import argparse
import importlib.metadata
import platform
import torch
from bscg.data import write_json

if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);a=p.parse_args()
    write_json(a.out,dict(python=platform.python_version(),platform=platform.platform(),
        packages={name:importlib.metadata.version(name) for name in ["torch","numpy","pandas"]},
        cuda=torch.version.cuda,cuda_available=torch.cuda.is_available(),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None))
