
<div style="position:fixed;top:10px;right:20px;z-index:999;">
  <a href="README_zh.md" style="background:#007bff;color:white;padding:6px 12px;border-radius:6px;text-decoration:none;">中文</a>
</div>



# Qualcomm-AI-Hub
Qualcomm AI Hub end-to-end model deployment tutorial: Supports cloud compilation for any PyTorch model, processor performance analysis, cloud inference verification, and exports directly deployable DLC models. Includes complete environment setup, model saving (torch.jit.trace), proxy configuration, common issue solutions, and reusable scripts. Works with various chips and input dimensions; beginners can get started with one click.

> Suitable for cloud compilation, processor performance testing, inference validation of any PyTorch model, and exporting DLC models deployable to various Qualcomm chips.
> Note: This tutorial uses SA8295P ADP chip and 38-dimensional input as examples; you can replace them with any chip and input dimension as needed.

---

## 1. Prerequisites
### 1.1 Register Qualcomm AI Hub Account
1. Open the official website: https://aihub.qualcomm.com/
2. Register and log in (free quota is sufficient for development and testing)
3. Go to Personal Center → **API Keys**
4. Create and copy your API Key (required for subsequent use)

### 1.2 Local Environment Requirements
- Python 3.9 ~ 3.11
- No local GPU required; CPU only
- Internet access (proxy configuration required)

---

## 2. Environment Installation
### 2.1 Create Virtual Environment (Recommended)
```bash
# Create
python -m venv qai_env
# Activate on Windows
qai_env\Scripts\activate
# Activate on Linux / macOS
source qai_env/bin/activate
```

### 2.2 Install Dependencies
```bash
pip install qai-hub
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy h5py
```

---

## 3. Configure Authentication and Proxy
### 3.1 Set QAI Hub API Key
**Method 1: Configure in code (Recommended)**
Add at the top of the script:
```python
import os
os.environ["QAI_HUB_API_KEY"] = "YOUR_API_KEY"
```

**Method 2: Temporary terminal configuration**
```bash
# Windows PowerShell
$env:QAI_HUB_API_KEY="YOUR_API_KEY"
# Linux / macOS
export QAI_HUB_API_KEY="YOUR_API_KEY"
```

### 3.2 Configure Proxy (Required in mainland China)
Set the port according to your proxy tool:
- Clash: 7890
- V2Ray / Others: 10809

```python
def setup_proxy(proxy_port: int = 10809):
    os.environ["http_proxy"] = f"http://127.0.0.1:{proxy_port}"
    os.environ["https_proxy"] = f"http://127.0.0.1:{proxy_port}"
setup_proxy(10809)
```

Note: The device `SA8295P ADP` in the tutorial is for demonstration only. Please select the corresponding device on the Qualcomm AI Hub official website according to your needs. The device list on the official website is continuously updated; please check the latest device information to avoid operation failures due to device name changes.

The image below shows the device selection interface of Qualcomm AI Hub official website, clearly displaying the list of available Qualcomm chip devices. Developers can filter target devices and copy device names for subsequent model compilation configuration.

<img width="1257" height="602" alt="image" src="https://github.com/user-attachments/assets/40a93e50-dfe0-4ece-9dd7-70ec8231ac80" />

---

## 4. Complete Running Script
Save the following code as `qai_hub_deploy.py` (universal name for any chip/input), and place your TorchScript model (.pt format) in the same directory.

### Supplement: Model Saving Requirement (Must use torch.jit.trace)
The model used in the tutorial must be saved as TorchScript format (.pt file) via `torch.jit.trace`, otherwise cloud compilation will fail. The specific saving code is as follows (replace with your model structure and input dimensions):

```python
import torch
from your_model import YourModel  # Replace with your model import path (e.g., SEResnet)

# 1. Initialize your model (same structure as training)
model = YourModel(...)  # Fill in your model parameters
model.eval()  # Must switch to evaluation mode

# 2. Generate dummy input matching actual input specifications (example: (1,1,38))
dummy_input = torch.randn(1, 1, 38)  # Replace with your input shape (e.g., (1,3,224,224))

# 3. Trace and save model with torch.jit.trace (core step)
traced_model = torch.jit.trace(model, dummy_input)
traced_model.save("traced_model.pt")  # Customize filename, modify accordingly in subsequent scripts

print("Model saved as TorchScript format via torch.jit.trace")
```

Note: Ensure the dummy input shape matches your actual input dimensions and the model is in eval mode during saving, otherwise cloud compilation will fail.

```python
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
# Proxy Setup
# ======================
def setup_proxy(proxy_port: int = 10809) -> None:
    os.environ["http_proxy"] = f"http://127.0.0.1:{proxy_port}"
    os.environ["https_proxy"] = f"http://127.0.0.1:{proxy_port}"
    print(f"Proxy set to 127.0.0.1:{proxy_port}")

setup_proxy(10809)  # Replace with your proxy port

# ======================
# Model Compilation
# ======================
def load_and_compile_model(model_path: str, target_device: str, input_shape: tuple):
    """
    Universal model compilation function
    :param model_path: Local TorchScript model path (.pt)
    :param target_device: Target Qualcomm chip (e.g., "SA8295P ADP", obtained from official website)
    :param input_shape: Model input shape (e.g., (1,1,38), replace with your dimensions)
    :return: Compiled target model
    """
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None
    try:
        print("Loading model...")
        model = torch.jit.load(model_path).eval()
        print(f"Submitting compile job (Target Device: {target_device})...")
        job = hub.submit_compile_job(
            model=model,
            device=hub.Device(target_device, os="14"),  # Adjust OS version per official website
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
   
    X = np.load("label_know_data.npz")["x"] # Load real data
    X = X.reshape((X.shape[0], 1, X.shape[1]))  
    input_array = X[0:1].astype(np.float32)      
    print(f"Using real data, shape: {input_array.shape}")
    
    print(f"Generated input data, shape: {input_array.shape}")
    
    try:
        print("Running inference...")
        inference_job = hub.submit_inference_job(
            model=model,
            device=model.device,  # Auto-match compiled target chip
            inputs=dict(image=[input_array]),
        )
        
        inference_job.wait()
        inference_out_dir = "inference_output"
        inference_job.download_results(inference_out_dir)
        print(f"Inference results saved to: {inference_out_dir}")
        
        # Recursively find Dataset in H5 file
        def get_h5_data(h5_item):
            if isinstance(h5_item, h5py.Dataset):
                return h5_item[:]
            for key in h5_item.keys():
                res = get_h5_data(h5_item[key])
                if res is not None:
                    return res
            return None
        
        # Read H5 result file
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
    print(f"【Real On-Chip Hardware Performance (Measured in Cloud)】:")
    print(f" -> Pure hardware inference latency: {chip_latency_ms:.4f} ms ({p50_us} us)")
    print(f" -> Chip NPU peak memory usage: {chip_mem_mb:.2f} MB")

# ======================
# Download Deployable DLC Model
# ======================
def download_deployable_model(target_model: hub.Model, save_name: str = "final_deploy_model.dlc") -> None:
    """Download DLC model file deployable to target Qualcomm chip (universal name)"""
    try:
        target_model.download(save_name)
        print(f"\n[Success] Task completed. Deployable DLC model saved as: {save_name}")
    except Exception as e:
        print(f"\n[Failed] Model download failed: {str(e)}")

# ======================
# Main Flow (Customize chip, input shape, model path)
# ======================
if __name__ == "__main__":
    # --------------------------
    # Customizable Parameters
    # --------------------------
    model_path = "SEResnet_model_traced_model.pt"  # Replace with your TorchScript model path
    target_device = "SA8295P ADP"  # Replace with target chip name from official website
    input_shape = (1, 1, 38)       # Replace with your model input shape (example: 38-dim input)
    
    # Run main pipeline
    target_model = load_and_compile_model(model_path, target_device, input_shape)
    if not target_model:
        exit(1)
    chip_latency_ms, chip_mem_mb, p50_us = profile_model(target_model, target_device)
    logits = run_inference(target_model, input_shape)
    generate_report(logits, chip_latency_ms, chip_mem_mb, p50_us, target_device)
    download_deployable_model(target_model)
```

Note: This script is universal. Modify the "Customizable Parameters" section to adapt to any Qualcomm chip and input dimension.
SA8295P ADP + 38-dimensional input is only an example; not limited to this configuration.

---

## 5. Run Command
```bash
python qai_hub_deploy.py
```

---

## 6. Performance Analysis
After the model is compiled on the target chip, the Qualcomm AI Hub official website will generate a performance analysis report with key indicators:
- Inference Time
- Memory Usage
- Compute Units

<img width="1278" height="529" alt="image" src="https://github.com/user-attachments/assets/5ffa0841-515a-4aad-955e-b12a67c7c7cf" />

---

## 7. Execution Flow Description
1. Disable local CUDA to avoid environment interference
2. Load TorchScript model saved via torch.jit.trace
3. Cloud-compile to DLC format dedicated to target chip (any Qualcomm chip can be replaced)
4. Perform performance profiling on real processor hardware (latency, memory test)
5. Perform cloud inference verification (adapt to any input dimension, verify normal model output)
6. Generate universal hardware performance report (auto-adapts to target chip)
7. Export `.dlc` model file that can be directly deployed to target chip

---

## 8. Output File Description
| File / Directory | Purpose |
|---|---|
| `profile_output/` | Hardware performance test JSON results (including detailed data such as latency and memory) |
| `inference_output/` | Cloud inference H5 output file (including model inference results) |
| `final_deploy_model.dlc` | Model file directly deployable to target Qualcomm chip |
| `qai_hub_deploy.py` | Complete deployment script (reusable, configurable for different scenarios) |
| `traced_model.pt` | TorchScript model saved via torch.jit.trace (needs to be generated by user) |

---

## 9. Common Issues
### 9.1 Connection Timeout / Cannot access Qualcomm AI Hub
- Wrong proxy port: Check proxy tool port (Clash=7890, V2Ray=10809), modify port in `setup_proxy` function
- Proxy not enabled: Ensure local proxy tool is running normally and can access external network
- API Key configuration error: Regenerate API Key, ensure no extra spaces or line breaks

### 9.2 Model Compilation Failed
- Wrong model format: Not saved via `torch.jit.trace`, re-run model saving code
- Input shape mismatch: Dummy input shape inconsistent with actual model input dimensions, modify input shape in model saving and script
- Wrong chip name: Target chip name inconsistent with official website, re-obtain from official website and modify `target_device` parameter
- Model corrupted: Re-export TorchScript model, ensure model is in eval mode

### 9.3 Performance Profiling / Inference Failed
- Insufficient directory permissions: Ensure local directory is readable and writable, avoid permission errors
- H5/JSON file not found: Wait for task to complete, do not terminate script midway
- OS version mismatch: Adjust `os` parameter in `hub.Device` according to official chip requirements (default os="14")

### 9.4 Model Download Failed
- Unstable network: Re-run script, execute `download_deployable_model` function separately
- Insufficient permissions: Modify DLC model saving path, ensure directory is writable

---

## 10. Contact Author
If you encounter any problems, need assistance or have suggestions during the use of this tutorial, script running or Qualcomm AI Hub deployment process:
- Email: ahufcy123@163.com
- Welcome to send emails to communicate technical problems, deployment errors, model adaptation solutions; I will reply as soon as possible.

