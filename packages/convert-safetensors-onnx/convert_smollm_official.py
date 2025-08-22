#!/usr/bin/env python3
import os
import json
import shutil
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig
from optimum.onnxruntime import ORTModelForCausalLM
from huggingface_hub import HfApi, create_repo
import traceback


def convert_original_smollm_to_onnx(source_repo="maximuspowers/smollm-convo-filler", target_repo="maximuspowers/smollm-convo-filler-onnx", local_path="./smollm-convo-filler-onnx-official"):        
    print(f"Converting {source_repo} to ONNX...")
    
    try:
        # rm if alr exists
        if os.path.exists(local_path):
            shutil.rmtree(local_path)
        os.makedirs(local_path)
        
        # optimum method for converting to onnx
        ort_model = ORTModelForCausalLM.from_pretrained(
            source_repo,
            export=True,
            provider="CPUExecutionProvider",
            use_io_binding=False,
        )
        tokenizer = AutoTokenizer.from_pretrained(source_repo)
        config = AutoConfig.from_pretrained(source_repo)
        
        # needs pad token set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("Saving ONNX model...")
        tokenizer.save_pretrained(local_path)
        config_dict = config.to_dict()
        config_dict.update({
            "torch_dtype": "float32",
            "use_cache": False, # required for transformers.js
        })
        
        with open(os.path.join(local_path, "config.json"), 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # transformers.js requires it in onnx subdir
        onnx_dir = os.path.join(local_path, "onnx")
        os.makedirs(onnx_dir, exist_ok=True)
        ort_model.save_pretrained(onnx_dir)
        
        print(f"ONNX conversion completed, saved to: {local_path}")
        
        return local_path, target_repo
        
    except Exception as e:
        print(f"ONNX conversion failed: {e}")
        traceback.print_exc()
        return None, None

def upload(local_path, repo_name):    
    print(f"Uploading model to {repo_name}")
    try:
        api = HfApi()
        create_repo(repo_name, exist_ok=True)
        api.upload_folder(
            folder_path=local_path,
            repo_id=repo_name,
            commit_message="onnx conversion"
        )
        print(f"Model uploaded to: https://huggingface.co/{repo_name}")
        return True
        
    except Exception as e:
        print(f"Upload failed: {e}")
        return False

if __name__ == "__main__":   
    local_path, target_repo = convert_original_smollm_to_onnx()
    upload(local_path, target_repo)
    