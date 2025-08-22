#!/usr/bin/env python3
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
        
        print(f"Python Input token length: {len(inputs['input_ids'][0])}")
        print(f"Python Input tokens (first 15): {inputs['input_ids'][0][:15].tolist()}")
        
        key_tokens = ["<|im_start|>", "<|im_end|>", "user", "knowledge", "<|sil|>"]
        for token in key_tokens:
            encoded = tokenizer.encode(token, add_special_tokens=False)
            print(f"Python '{token}' -> {encoded}")
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **generation_params
            )
        end_time = time.time()
        
        print(f"Output token length: {len(outputs[0])}")
        
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        print(f"Full decoded response: {repr(full_response)}")
        
        input_length = len(inputs['input_ids'][0])
        if len(outputs[0]) > input_length:
            new_tokens = outputs[0][input_length:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        else:
            response = ""
        
        return response, end_time - start_time

    def compare_responses(self, user_input, **generation_params):
        context_prompt = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>knowledge\n<|sil|><|im_end|>\n"
        
        print(f"User Input: {repr(user_input)}")
        print(f"Formatted Prompt: {repr(context_prompt)}")
        print("=" * 80)
        
        original_response, original_time = self.generate_response(
            self.original_model, self.original_tokenizer, context_prompt, **generation_params
        )
        
        onnx_response, onnx_time = self.generate_response(
            self.onnx_model, self.onnx_tokenizer, context_prompt, **generation_params
        )
        
        import re
        original_cleaned = original_response.replace("<|im_start|>", "").replace("<|im_end|>", "")
        original_cleaned = re.sub(r'^assistant\s*', '', original_cleaned, flags=re.IGNORECASE)
        original_cleaned = original_cleaned.strip().split("\n")[0]
        
        onnx_cleaned = onnx_response.replace("<|im_start|>", "").replace("<|im_end|>", "")
        onnx_cleaned = re.sub(r'^assistant\s*', '', onnx_cleaned, flags=re.IGNORECASE) 
        onnx_cleaned = onnx_cleaned.strip().split("\n")[0]
        
        print(f"Cleaning steps for original:")
        step1 = original_response.replace('<|im_start|>', '').replace('<|im_end|>', '')
        step2 = re.sub(r'^assistant\s*', '', step1, flags=re.IGNORECASE)
        print(f"  1. After removing ChatML: {repr(step1)}")
        print(f"  2. After removing 'assistant': {repr(step2)}")
        print(f"  3. After trim + split[0]: {repr(original_cleaned)}")
        print()
        
        print(f"Original (SafeTensors) [{original_time:.3f}s]:")
        print(f"   Raw: {repr(original_response)}")
        print(f"   Cleaned: {repr(original_cleaned)}")
        print()
        print(f"ONNX Converted [{onnx_time:.3f}s]:")
        print(f"   Raw: {repr(onnx_response)}")
        print(f"   Cleaned: {repr(onnx_cleaned)}")
        print()
        
        exact_match = onnx_cleaned == original_cleaned
        print(f"Exact match (cleaned): {exact_match}")
        
        if not exact_match:
            orig_tokens = set(original_cleaned.split())
            onnx_tokens = set(onnx_cleaned.split())
            
            if orig_tokens or onnx_tokens:
                jaccard_sim = len(orig_tokens & onnx_tokens) / len(orig_tokens | onnx_tokens)
                print(f"Token similarity (Jaccard): {jaccard_sim:.3f}")
            
            print(f"Length difference: {len(onnx_cleaned) - len(original_cleaned)} chars")
        
        print(f"Speed ratio (ONNX/Original): {onnx_time / original_time:.2f}x")
        print("=" * 80)
        print()
        
        return {
            'user_input': user_input,
            'formatted_prompt': context_prompt,
            'original_response': original_response,
            'onnx_response': onnx_response,
            'original_cleaned': original_cleaned,
            'onnx_cleaned': onnx_cleaned,
            'exact_match': exact_match,
            'original_time': original_time,
            'onnx_time': onnx_time
        }

    def run_comprehensive_test(self):
        print("Starting Model Comparison Test")
        print("=" * 80)
        print()
        
        test_prompts = [
            "I want to plan a trip to Seattle. Where do I start?",
        ]
        
        test_configs = [
            {
                "max_new_tokens": 128,
                "temperature": 1.0,
                "do_sample": False,  
            }
        ]
        
        results = []
        
        for config_idx, config in enumerate(test_configs):
            print(f"🔧 Test Configuration {config_idx + 1}: {config}")
            print("-" * 40)
            
            for prompt_idx, user_input in enumerate(test_prompts):
                print(f"Test {config_idx + 1}.{prompt_idx + 1}:")
                result = self.compare_responses(user_input, **config)
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
        comparator.run_comprehensive_test()
        
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