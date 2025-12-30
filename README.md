# vllm-sim-profiler

```bash
export HF_TOKEN="<YOUR HF TOKEN>"
kubectl create secret generic hf-secret --from-literal="HF_TOKEN=${HF_TOKEN}"

k apply -f pvc-gke.yaml
k apply -f debug-pod.yaml
```