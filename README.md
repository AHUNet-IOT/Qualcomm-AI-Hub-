<div style="position:fixed;top:10px;right:20px;z-index:999;">
  <a href="README_zh.md" style="background:#007bff;color:white;padding:6px 12px;border-radius:6px;text-decoration:none;">中文</a>
</div>

# Qualcomm AI Hub: End-to-End Model Deployment Tutorial

# Qualcomm-AI-Hub

End-to-end model deployment tutorial for Qualcomm AI Hub: supports cloud compilation of any PyTorch model, processor performance analysis, cloud inference verification, and exports DLC models ready for direct deployment. Includes complete environment configuration, model saving (torch.jit.trace), proxy settings, common problem solutions and reusable scripts. Suitable for different chips and input dimensions, beginners can get started with one click.

Suitable for cloud compilation of any PyTorch model, processor performance testing, inference verification, and export DLC models deployable to various Qualcomm chips

Note: The tutorial takes SA8295P ADP chip and 38-dimensional input as examples, you can replace them with any chip and input dimension according to your needs

---

## 1. Preparations

### 1.1 Register Qualcomm AI Hub Account

1. Open the official website: [https://aihub.qualcomm.com/](https://aihub.qualcomm.com/)

2. Register and log in to your account (free quota is enough for development and testing)

3. Enter Personal Center → **API Keys**

4. Create and copy your API Key (must be used later)

### 1.2 Local Environment Requirements

- Python 3.9 ~ 3.11

- No GPU required locally, CPU is enough

- Access to the external network (proxy configuration required)

---

## 2. Environment Installation

### 2.1 Create Virtual Environment (Recommended)

```Bash

# Create
python -m venv qai_env
# Activate on Windows
qai_env\Scripts\activate
# Activate on Linux / macOS
source qai_env/bin/activate
```

### 2.2 Install Dependencies

```Bash

pip install qai-hub
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy h5py
```

---

## 3. Configure Authentication and Proxy

### 3.1 Set QAI Hub API Key

**Method 1: Configure in Code (Recommended)**

Add at the top of the script:

```Python

import os
os.environ["QAI_HUB_API_KEY"] = "YOUR_API_KEY"
```

**Method 2: Temporary Configuration in Terminal**

```Bash

# Windows PowerShell
$env:QAI_HUB_API_KEY="YOUR_API_KEY"
# Linux / macOS
export QAI_HUB_API_KEY="YOUR_API_KEY"
```

### 3.2 Configure Proxy (Required for Mainland China)

Fill in the port according to your proxy tool:

- Clash：7890

- V2Ray / Other：10809

```Python

def setup_proxy(proxy_port: int = 10809):
    os.environ["http_proxy"] = f"http://127.0.0.1:{proxy_port}"
    os.environ["https_proxy"] = f"http://127.0.0.1:{proxy_port}"
setup_proxy(10809)
```

Note: The device `SA8295P ADP` used in the tutorial is only an example. Please select the corresponding device on the Qualcomm AI Hub official website according to your needs. The device list on the official website will be updated continuously with versions, please pay attention to the latest device information in time to avoid operation failure due to device name changes.

---

## 4. Complete Running Script

Save the following code as `qai_hub_deploy.py` (universal name, suitable for any chip/input), and place your TorchScript model (.pt format) in the same directory.

### Supplement: Model Saving Requirements (Must use torch.jit.trace)

The model used in the tutorial needs to be saved as TorchScript format (.pt file) through `torch.jit.trace` tracking, otherwise cloud compilation will fail. The specific saving code is as follows (replace with your model structure and input dimensions):

```Python

import torch
from your_model import YourModel
# 1. Initialize your model (same structure as training)
model = YourModel(...)
model.eval()
# 2. Generate dummy input matching actual input size
dummy_input = torch.randn(1, 1, 38)
# 3. Trace and save model
traced_model = torch.jit.trace(model, dummy_input)
traced_model.save("traced_model.pt")
print("Model saved as TorchScript via torch.jit.trace")
```

```Python

# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TORCH_CUDA_VERSION"] = ""
import torch
torch.cuda.is_available = lambda: False
torch.backends.cudnn.enabled = False
import numpy as np
import json
import glob
import h5py
import qai_hub as hub

# ======================
# Proxy Settings
# ======================
def setup_proxy(proxy_port: int = 10809) -> None:
    os.environ["http_proxy"] = f"http://127.0.0.1:{proxy_port}"
    os.environ["https_proxy"] = f"http://127.0.0.1:{proxy_port}"
    print(f"Proxy set to 127.0.0.1:{proxy_port}")
setup_proxy(10809)

# ======================
# Model Compilation
# ======================
def load_and_compile_model(model_path: str, target_device: str, input_shape: tuple):
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None
    try:
        print("Loading model...")
        model = torch.jit.load(model_path).eval()
        print(f"Submitting compile job (Target Device: {target_device})...")
        job = hub.submit_compile_job(
            model=model,
            device=hub.Device(target_device, os="14"),
            input_specs=dict(image=input_shape),
            options="--target_runtime qnn_dlc",
        )
        job.wait()
        target_model = job.get_target_model()
        print("Compile success.")
        return target_model
    except Exception as e:
        print(f"Compile failed: {e}")
        return None

# ======================
# Performance Profiling
# ======================
def profile_model(model, target_device: str):
    try:
        print(f"Start profiling (Target Device: {target_device})...")
        job = hub.submit_profile_job(
            model=model,
            device=hub.Device(target_device, os="14"),
        )
        job.wait()
        out_dir = "profile_output"
        job.download_results(out_dir)
        js = glob.glob(f"{out_dir}/*.json")[0]
        with open(js) as f:
            data = json.load(f)
        s = data["execution_summary"]
        p50_us = s["estimated_inference_time"]
        mem = s["estimated_inference_peak_memory"]
        ms = p50_us / 1000
        mb = mem / 1024 / 1024
        print(f"Latency: {ms:.4f}ms  Memory: {mb:.2f}MB")
        return ms, mb, p50_us
    except Exception as e:
        print(f"Profile failed: {e}")
        return 0, 0, 0

# ======================
# Cloud Inference
# ======================
def run_inference(model, input_shape: tuple):
    input_array = np.random.rand(*input_shape).astype(np.float32)
    print(f"Generated input data, shape: {input_array.shape}")
    
    try:
        print("Running inference...")
        inference_job = hub.submit_inference_job(
            model=model,
            device=model.device,
            inputs=dict(image=[input_array]),
        )
        
        inference_job.wait()
        inference_out_dir = "inference_output"
        inference_job.download_results(inference_out_dir)
        print(f"Inference results saved to: {inference_out_dir}")
        
        def get_h5_data(h5_item):
            if isinstance(h5_item, h5py.Dataset):
                return h5_item[:]
            for key in h5_item.keys():
                res = get_h5_data(h5_item[key])
                if res is not None:
                    return res
            return None
        
        h5_files = glob.glob(os.path.join(inference_out_dir, "*.h5"))
        if not h5_files:
            print("Error: No inference H5 file found")
            return None
        
        with h5py.File(h5_files[0], 'r') as f:
            logits = get_h5_data(f)
        
        print(f"Inference completed, output logits shape: {logits.shape}")
        return logits
    
    except Exception as e:
        print(f"Inference failed: {str(e)}")
        return None

# ======================
# Generate Report
# ======================
def generate_report(logits, chip_latency_ms: float, chip_mem_mb: float, p50_us: int, target_device: str):
    if logits is None:
        print("Failed to generate report: Inference results are empty")
        return
    
    print(f"             Qualcomm AI Hub Model Performance Report (Target Device: {target_device})")
    print("=" * 65)
    print(f"【Real Hardware Performance on Chip (Measured in Cloud)】:")
    print(f" -> Inference Latency: {chip_latency_ms:.4f} ms ({p50_us} us)")
    print(f" -> NPU Peak Memory: {chip_mem_mb:.2f} MB")

# ======================
# Download Deployable DLC Model
# ======================
def download_deployable_model(target_model: hub.Model, save_name: str = "final_deploy_model.dlc") -> None:
    try:
        target_model.download(save_name)
        print(f"\n[Success] Task completed. Deployable DLC model saved as: {save_name}")
    except Exception as e:
        print(f"\n[Failed] Model download failed: {str(e)}")

# ======================
# Main Workflow
# ======================
if __name__ == "__main__":
    model_path = "SEResnet_model_traced_model.pt"
    target_device = "SA8295P ADP"
    input_shape = (1, 1, 38)
    
    target_model = load_and_compile_model(model_path, target_device, input_shape)
    if not target_model:
        exit(1)
    chip_latency_ms, chip_mem_mb, p50_us = profile_model(target_model, target_device)
    logits = run_inference(target_model, input_shape)
    generate_report(logits, chip_latency_ms, chip_mem_mb, p50_us, target_device)
    download_deployable_model(target_model)
```

---

## 5. Run Command

```Bash

python qai_hub_deploy.py
```

---

## 6. Performance Analysis

After the model is compiled on the target chip, the Qualcomm AI Hub official website will generate a performance analysis report with key indicators:

- Inference Time

- Memory Usage

- Compute Units

---

## 7. Execution Flow

1. Disable local CUDA to avoid environment interference

2. Load TorchScript model saved by torch.jit.trace

3. Cloud compile to DLC format for target chip

4. Run performance profiling on real processor hardware

5. Perform cloud inference verification

6. Generate hardware performance report

7. Export .dlc model for direct deployment

---

## 8. Output Files Description

|File/Directory|Usage|
|---|---|
|`profile_output/`|Hardware profiling JSON results|
|`inference_output/`|Cloud inference H5 output|
|`final_deploy_model.dlc`|Deployable model for target Qualcomm chip|
|`qai_hub_deploy.py`|Full deployment script|
|`traced_model.pt`|TorchScript model saved via torch.jit.trace|
---

## 9. Common Issues

### 9.1 Connection Timeout / Cannot Access Qualcomm AI Hub

- Wrong proxy port: Check proxy port (Clash=7890, V2Ray=10809)

- Proxy not enabled: Ensure proxy is running

- API Key error: Regenerate API Key

### 9.2 Model Compilation Failed

- Wrong model format: Use torch.jit.trace to save

- Input shape mismatch: Match dummy input with real input

- Wrong chip name: Use exact name from official website

- Model corrupted: Re-export model in eval mode

### 9.3 Profiling / Inference Failed

- Insufficient permissions: Ensure directory writable

- H5/JSON not found: Wait for job completion

- OS version mismatch: Adjust os parameter in Device

### 9.4 Model Download Failed

- Unstable network: Re-run download function

- Insufficient permissions: Change save path

---

## 10. Contact Author

- Email: [ahufcy123@163.com](mailto:ahufcy123@163.com)

- Welcome to contact for technical issues, deployment errors, model adaptation solutions
