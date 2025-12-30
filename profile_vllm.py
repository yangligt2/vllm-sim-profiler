import time
import pandas as pd
import itertools
from vllm import EngineArgs, LLMEngine, SamplingParams

# --- Configuration ---
# Use the internal model path often found in vllm containers or your specific model path
# If you are in the container, you might need to point to where the model is mounted (e.g., /model)
# or use a HuggingFace Hub ID.
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"

# Adjust this to match your GPU count (e.g. 8 for your 8xGPU server)
TP_SIZE = 4

MAX_MODEL_LEN = 32000
ENABLE_CHUNKED_PREFILL = False
CSV_FILENAME = "vllm_profiling_data.csv"

# Standard Python counter to replace the vllm.utils one
request_id_counter = itertools.count()

def initialize_engine():
    engine_args = EngineArgs(
        model=MODEL_NAME,
        tensor_parallel_size=TP_SIZE,
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=False,
        enable_chunked_prefill=ENABLE_CHUNKED_PREFILL,
        disable_log_stats=True,
        trust_remote_code=True # Often needed for some models
    )
    return LLMEngine.from_engine_args(engine_args)

def generate_dummy_prompt(token_len):
    # Generates a string that roughly tokenizes to token_len
    # Approximation: 1 token ~= 4 chars, but we use a repeating pattern to be safe
    return "benchmark " * (token_len // 2)

def warmup_gpu(engine):
    print(f">>> Warming up GPU for 20 seconds...")
    dummy_prompt = generate_dummy_prompt(1024)
    start_time = time.time()
    while time.time() - start_time < 20:
        if not engine.has_unfinished_requests():
             for _ in range(16):
                rid = f"warmup_{next(request_id_counter)}"
                engine.add_request(rid, dummy_prompt, SamplingParams(max_tokens=20, ignore_eos=True))
        engine.step()
    while engine.has_unfinished_requests():
        engine.step()
    print(">>> Warmup complete.")

def run_profiling():
    try:
        engine = initialize_engine()
    except Exception as e:
        print(f"Failed to initialize engine: {e}")
        return

    print(">>> Engine Initialized. Starting Profiling...")

    results = []

    # ==========================================
    # Phase 1: Prefill Sweep (Isolating Beta_1)
    # ==========================================
    print("\n--- Phase 1: Prefill Sweep ---")
    prefill_lengths = [128, 512, 1024, 2048, 4096, 8000, 16000, 30000]

    for length in prefill_lengths:
        prompt = generate_dummy_prompt(length)
        req_id = f"prefill_{length}_{next(request_id_counter)}"

        engine.add_request(
            req_id,
            prompt,
            SamplingParams(max_tokens=1, ignore_eos=True)
        )

        # Measure Step Time (First step is Prefill)
        start_time = time.perf_counter()
        request_outputs = engine.step()
        end_time = time.perf_counter()

        if request_outputs:
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

        # Clean up
        while engine.has_unfinished_requests():
            engine.step()

    # ==========================================
    # Phase 2: Decode Sweep (Isolating Beta_2, Beta_3)
    # ==========================================
    print("\n--- Phase 2: Decode Sweep ---")
    batch_sizes = [1, 16, 32, 64, 128]
    # Ensure these fit in your GPU memory with the model loaded
    context_lengths = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 30000]

    for ctx_len in context_lengths:
        for batch_size in batch_sizes:
            req_ids = []
            prompt = generate_dummy_prompt(ctx_len)

            # 1. Warmup / Setup Batch
            for i in range(batch_size):
                rid = f"decode_b{batch_size}_l{ctx_len}_{next(request_id_counter)}"
                req_ids.append(rid)
                engine.add_request(rid, prompt, SamplingParams(max_tokens=10, ignore_eos=True))

            # 2. Force Prefill
            finished_prefill_ids = set()
            while len(finished_prefill_ids) < batch_size:
                outputs = engine.step()
                if not outputs:
                    # If outputs is empty but we haven't finished all requests, 
                    # it implies some requests were aborted or failed. Break to avoid infinite loop.
                    if not engine.has_unfinished_requests():
                        break
                    continue
                
                # Check outputs from this step
                for out in outputs:
                    if out.request_id in req_ids:
                        # If a request has output tokens, it has finished prefill
                        if len(out.outputs) > 0:
                            finished_prefill_ids.add(out.request_id)

            # 3. Measure Pure Decode Steps
            steps_to_measure = 20
            for _ in range(steps_to_measure):
                # Calculate "Potentially" total usage (for reference)
                requested_kv_usage = batch_size * ctx_len

                start_time = time.perf_counter()
                outputs = engine.step()  # <--- Capture the output!
                end_time = time.perf_counter()
                
                # --- CRITICAL FIX: Measure what ACTUALLY ran ---
                actual_num_reqs = len(outputs)
                # Sum of context lengths of the requests that actually ran
                actual_total_kv = sum([len(o.prompt_token_ids) + len(o.outputs) for o in outputs])

                duration_ms = (end_time - start_time) * 1000

                results.append({
                    "type": "decode",
                    "num_prefill_tokens": 0,
                    "num_decode_reqs": actual_num_reqs,  # Log actual
                    "total_kv_usage": actual_total_kv,   # Log actual
                    "duration_ms": duration_ms
                })

            print(f"Decode  | Batch: {batch_size} | KV Total: {batch_size*ctx_len} | Avg Time: {duration_ms:.2f}ms")

            # 4. Cleanup
            for rid in req_ids:
                try:
                    engine.abort_request(rid)
                except KeyError:
                    pass
            while engine.has_unfinished_requests():
                engine.step()

    df = pd.DataFrame(results)
    df.to_csv(CSV_FILENAME, index=False)
    print(f"\n>>> Data saved to {CSV_FILENAME}")

if __name__ == "__main__":
    run_profiling()