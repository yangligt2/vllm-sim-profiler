import time
import pandas as pd
import numpy as np
from vllm import EngineArgs, LLMEngine, SamplingParams, RequestOutput
from vllm.utils import Counter

# --- Configuration ---
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct" # Change this to your model path
TP_SIZE = 1 # Tensor Parallelism (Set to match your target config, e.g., 4 or 8)
MAX_MODEL_LEN = 8192
# We disable chunked prefill to get clean "linear" prefill measurements for calibration
ENABLE_CHUNKED_PREFILL = False 

# Output file
CSV_FILENAME = "vllm_profiling_data.csv"

def initialize_engine():
    engine_args = EngineArgs(
        model=MODEL_NAME,
        tensor_parallel_size=TP_SIZE,
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=False, # True ensures less CUDA graph noise, False is realistic
        enable_chunked_prefill=ENABLE_CHUNKED_PREFILL,
        disable_log_stats=True
    )
    return LLMEngine.from_engine_args(engine_args)

def generate_dummy_prompt(token_len):
    # Generates a string that roughly tokenizes to token_len
    # (Approximation: 1 token ~= 4 chars)
    return "benchmark " * (token_len // 2)

def run_profiling():
    engine = initialize_engine()
    print(">>> Engine Initialized. Starting Profiling...")
    
    results = []
    request_id_counter = Counter()

    # ==========================================
    # Phase 1: Prefill Sweep (Isolating Beta_1)
    # ==========================================
    print("\n--- Phase 1: Prefill Sweep ---")
    prefill_lengths = [128, 512, 1024, 2048, 4096, 6000]
    
    for length in prefill_lengths:
        prompt = generate_dummy_prompt(length)
        req_id = f"prefill_{length}_{next(request_id_counter)}"
        
        # Add request
        engine.add_request(
            req_id, 
            prompt, 
            SamplingParams(max_tokens=1, ignore_eos=True)
        )
        
        # Measure Step Time (First step is Prefill)
        start_time = time.perf_counter()
        request_outputs = engine.step()
        end_time = time.perf_counter()
        
        # Calculate tokens processed
        # Note: request_outputs[0].prompt_token_ids gives exact length
        actual_len = len(request_outputs[0].prompt_token_ids)
        duration_ms = (end_time - start_time) * 1000
        
        print(f"Prefill | Len: {actual_len} | Time: {duration_ms:.2f}ms")
        
        results.append({
            "type": "prefill",
            "num_prefill_tokens": actual_len,
            "num_decode_reqs": 0,
            "total_kv_usage": 0,
            "duration_ms": duration_ms
        })
        
        # Clean up (Finish the request)
        while engine.has_unfinished_requests():
            engine.step()

    # ==========================================
    # Phase 2: Decode Sweep (Isolating Beta_2, Beta_3)
    # ==========================================
    print("\n--- Phase 2: Decode Sweep ---")
    # Grid search: varying batch size and context length
    batch_sizes = [1, 8, 32, 64] 
    context_lengths = [128, 2048, 4096, 7000] # High values to stress Memory
    
    for ctx_len in context_lengths:
        for batch_size in batch_sizes:
            req_ids = []
            prompt = generate_dummy_prompt(ctx_len)
            
            # 1. Warmup / Setup Batch
            for i in range(batch_size):
                rid = f"decode_b{batch_size}_l{ctx_len}_{next(request_id_counter)}"
                req_ids.append(rid)
                engine.add_request(rid, prompt, SamplingParams(max_tokens=10, ignore_eos=True))
            
            # 2. Force Prefill (Wait until all requests are past prefill stage)
            # We don't record these times.
            while True:
                stats = engine.get_model_executor_guided_decoding_stats() # Just checking state
                outputs = engine.step()
                # Check if all requests have generated at least 1 token
                all_decoding = all(len(out.outputs) > 0 for out in outputs)
                if all_decoding and len(outputs) == batch_size:
                    break
            
            # 3. Measure Pure Decode Steps
            # We run a few steps to get a stable average
            steps_to_measure = 5
            for _ in range(steps_to_measure):
                
                # Capture accurate KV usage before step
                # Total KV = BatchSize * (ContextLen + GeneratedSoFar)
                # We approximate using the context_len as dominant factor
                current_kv_usage = batch_size * ctx_len 
                
                start_time = time.perf_counter()
                engine.step()
                end_time = time.perf_counter()
                
                duration_ms = (end_time - start_time) * 1000
                
                results.append({
                    "type": "decode",
                    "num_prefill_tokens": 0,
                    "num_decode_reqs": batch_size,
                    "total_kv_usage": current_kv_usage,
                    "duration_ms": duration_ms
                })
            
            print(f"Decode  | Batch: {batch_size} | KV Total: {batch_size*ctx_len} | Avg Time: {duration_ms:.2f}ms")

            # 4. Cleanup this batch
            for rid in req_ids:
                engine.abort_request(rid)
            while engine.has_unfinished_requests():
                engine.step()

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(CSV_FILENAME, index=False)
    print(f"\n>>> Data saved to {CSV_FILENAME}")

if __name__ == "__main__":
    run_profiling()