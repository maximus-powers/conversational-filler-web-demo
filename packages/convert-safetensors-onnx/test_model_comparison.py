#!/usr/bin/env python3
"""
Side-by-side comparison test between original SafeTensors model and converted ONNX model.
Tests multiple prompts to ensure conversion accuracy.
"""

import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from optimum.onnxruntime import ORTModelForCausalLM
import time
from datetime import datetime

class ModelComparator:
    def __init__(self, original_repo="maximuspowers/smollm-convo-filler", onnx_repo="maximuspowers/smollm-convo-filler-onnx-official"):
        self.original_repo = original_repo
        self.onnx_repo = onnx_repo
        
        print("Loading original SafeTensors model...")
        self.original_model = AutoModelForCausalLM.from_pretrained(original_repo)
        self.original_tokenizer = AutoTokenizer.from_pretrained(original_repo)
        
        print("Loading ONNX model from HuggingFace Hub...")
        self.onnx_model = ORTModelForCausalLM.from_pretrained(onnx_repo, subfolder="onnx")
        self.onnx_tokenizer = AutoTokenizer.from_pretrained(onnx_repo)
        
        if self.original_tokenizer.pad_token is None:
            self.original_tokenizer.pad_token = self.original_tokenizer.eos_token
        if self.onnx_tokenizer.pad_token is None:
            self.onnx_tokenizer.pad_token = self.onnx_tokenizer.eos_token
            
        print("Both models loaded successfully!\n")

    def generate_response(self, model, tokenizer, prompt, **generation_params):
        inputs = tokenizer(prompt, return_tensors="pt")
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                pad_token_id=tokenizer.pad_token_id,
                **generation_params
            )
        end_time = time.time()
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        
        return response, end_time - start_time

    def compare_responses(self, prompt, **generation_params):
        print(f"Prompt: {repr(prompt)}")
        print("=" * 80)
        
        original_response, original_time = self.generate_response(
            self.original_model, self.original_tokenizer, prompt, **generation_params
        )
        
        onnx_response, onnx_time = self.generate_response(
            self.onnx_model, self.onnx_tokenizer, prompt, **generation_params
        )
        
        print(f"Original (SafeTensors) [{original_time:.3f}s]:")
        print(f"   {repr(original_response)}")
        print()
        print(f"ONNX Converted [{onnx_time:.3f}s]:")
        print(f"   {repr(onnx_response)}")
        print()
        
        exact_match = original_response == onnx_response
        print(f"Exact match: {exact_match}")
        
        if not exact_match:
            orig_tokens = set(original_response.split())
            onnx_tokens = set(onnx_response.split())
            
            if orig_tokens or onnx_tokens:
                jaccard_sim = len(orig_tokens & onnx_tokens) / len(orig_tokens | onnx_tokens)
                print(f"Token similarity (Jaccard): {jaccard_sim:.3f}")
            
            print(f"Length difference: {len(onnx_response) - len(original_response)} chars")
        
        print(f"Speed ratio (ONNX/Original): {onnx_time / original_time:.2f}x")
        print("=" * 80)
        print()
        
        return {
            'prompt': prompt,
            'original_response': original_response,
            'onnx_response': onnx_response,
            'exact_match': exact_match,
            'original_time': original_time,
            'onnx_time': onnx_time
        }

    def run_comprehensive_test(self):
        print(f"Starting Model Comparison Test")
        print("=" * 80)
        print()
        
        test_prompts = [
            "Q: Hello, how are you?\nA:",
            "Q: What is the capital of France?\nA:",
            "Q: Explain quantum physics in simple terms.\nA:",
            "Q: How do I make a good pizza?\nA:",
            "Q: What's the weather like today?\nA:",
        ]
        
        test_configs = [
            {
                "max_new_tokens": 20,
                "do_sample": False, 
                "temperature": 1.0,
            },
            {
                "max_new_tokens": 50,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
            }
        ]
        
        results = []
        
        for config_idx, config in enumerate(test_configs):
            print(f"🔧 Test Configuration {config_idx + 1}: {config}")
            print("-" * 40)
            
            for prompt_idx, prompt in enumerate(test_prompts):
                print(f"Test {config_idx + 1}.{prompt_idx + 1}:")
                result = self.compare_responses(prompt, **config)
                results.append({**result, 'config': config})
        
        total_tests = len(results)
        exact_matches = sum(1 for r in results if r['exact_match'])
        
        print("SUMMARY STATISTICS")
        print("=" * 80)
        print(f"Total tests: {total_tests}")
        print(f"Exact matches: {exact_matches}/{total_tests} ({exact_matches/total_tests*100:.1f}%)")
        
        avg_speedup = np.mean([r['onnx_time'] / r['original_time'] for r in results])
        print(f"\n Average speed ratio (ONNX/Original): {avg_speedup:.2f}x")
        
        if avg_speedup < 1.0:
            print("ONNX model is faster on average")
        else:
            print("Original model is faster on average")
            
        return results

def main():
    """Main function to run the comparison test"""
    try:
        comparator = ModelComparator()
        results = comparator.run_comprehensive_test()
        
        print("\nComparison test completed successfully!")
        
    except Exception as e:
        if "not found" in str(e).lower() or "does not exist" in str(e).lower():
            print(f"❌ Error: {e}")
            print("Please ensure both models exist on HuggingFace Hub:")
            print("- maximuspowers/smollm-convo-filler")
            print("- maximuspowers/smollm-convo-filler-onnx-official")
        else:
            print(f"❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()