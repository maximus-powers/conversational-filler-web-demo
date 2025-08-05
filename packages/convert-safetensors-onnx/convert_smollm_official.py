#!/usr/bin/env python3
"""
Convert maximuspowers/smollm-convo-filler to ONNX using the official Hugging Face method.
Based on https://huggingface.co/blog/convert-transformers-to-onnx
"""

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

def test_onnx_model(local_path):    
    print("Testing model locally...")
    
    try:
        onnx_dir = os.path.join(local_path, "onnx")
        model = ORTModelForCausalLM.from_pretrained(onnx_dir, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
        test_prompt = "Q: Hello, how are you?\nA:"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Test gen successful")
        print(f"Test response: {response}")
        return True
        
    except Exception as e:
        print(f"Local test failed: {e}")
        traceback.print_exc()
        return False

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
    
    if local_path and target_repo:
        if test_onnx_model(local_path):           
            upload_choice = input("\nUpload to HF? (y/n): ").lower().strip()
            
            if upload_choice == 'y':
                if upload(local_path, target_repo):
                    print("Completed")
                else:
                    print("\nConversion successful but upload failed")
                    print(f"Local files at: {local_path}")
            else:
                print("\nONNX conversion completed locally")
                print(f"Files at: {local_path}")
        else:
            print("\nConversion completed but tests failed")
    else:
        print("\nONNX conversion failed")
