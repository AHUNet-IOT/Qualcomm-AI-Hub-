<div style="position:fixed;top:10px;right:20px;z-index:999;">
  <a href="README.md" style="background:#007bff;color:white;padding:6px 12px;border-radius:6px;text-decoration:none;">English</a>
</div>

# Qualcomm-AI-Hub
Qualcomm AI Hub 端到端模型部署教程：支持任意 PyTorch 模型云端编译、处理器性能分析、云端推理验证，并导出可直接部署的 DLC 模型。包含完整环境配置、模型保存（torch.jit.trace）、代理设置、常见问题解决方案与可复用脚本。适用于不同芯片与输入维度，新手可一键上手。

> 适用于任意 PyTorch 模型云端编译、处理器性能测试、推理验证，并导出可部署到各类高通芯片的 DLC 模型

> 注：教程中以 SA8295P ADP 芯片、24维输入为例，可根据自身需求替换为任意芯片与输入维度（24维输入：指数据集的单样本特征维度为 24，即每个输入样本包含 24 个特征数值）

---

## 1. 前置准备

### 1.1 注册 Qualcomm AI Hub 账号

1. 打开官网：https://aihub.qualcomm.com/

2. 注册并登录账号（免费额度可满足开发测试）

3. 进入个人中心 → **API Keys**

4. 创建并复制你的 API Key（后续必须使用）

### 1.2 本地环境要求

- Python 3.9 ~ 3.11

- 本地无需 GPU，CPU 即可

- 可访问外网（需配置代理）

---

## 2. 环境安装

### 2.1 创建虚拟环境（推荐）

```bash
# 创建
python -m venv qai_env

# Windows 激活
qai_env\Scripts\activate

# Linux / macOS 激活
source qai_env/bin/activate
```

### 2.2 安装依赖

```bash
pip install qai-hub
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy h5py
```

---

## 3. 配置认证与代理

### 3.1 设置 QAI Hub API Key

**方式 1：代码内配置（推荐）**

在脚本最顶部添加：

```python
import os
os.environ["QAI_HUB_API_KEY"] = "你的 API Key"
```

**方式 2：终端临时配置**

```bash
# Windows PowerShell
$env:QAI_HUB_API_KEY="你的 API Key"

# Linux / macOS
export QAI_HUB_API_KEY="你的 API Key"
```

### 3.2 配置代理（国内必配）

根据你的代理工具填写端口：

- Clash：7890

- V2Ray / 其他：10809



```python
def setup_proxy(proxy_port: int = 10809):
    os.environ["http_proxy"] = f"http://127.0.0.1:{proxy_port}"
    os.environ["https_proxy"] = f"http://127.0.0.1:{proxy_port}"

setup_proxy(10809)
```


注意：教程中使用的设备 `SA8295P ADP` 仅为示例，实际需根据自身需求在 Qualcomm AI Hub 官网上选择对应设备。官网设备列表会随版本更新不断迭代，需及时关注官网最新设备信息，避免因设备名称变更导致操作失败。

下图为 Qualcomm AI Hub 官网设备选择界面截图，清晰展示了可选择的各类高通芯片设备列表，开发者可在此界面筛选目标设备、复制设备名称，用于后续模型编译配置。

<img width="1257" height="602" alt="image" src="https://github.com/user-attachments/assets/40a93e50-dfe0-4ece-9dd7-70ec8231ac80" />


---

## 4. 完整运行脚本

将以下代码保存为 `qai_hub_deploy.py`（通用命名，适配任意芯片/输入），并将你的 TorchScript 模型（.pt格式）放在同一目录。

### 补充：模型保存要求（必须用 torch.jit.trace 形式）

教程中使用的模型需通过 `torch.jit.trace` 追踪保存为 TorchScript 格式（.pt文件），否则会导致云端编译失败，具体保存代码如下（替换为你的模型结构和输入维度）：

```python
import torch
from your_model import YourModel  # 替换为你的模型导入路径（如 SEResnet）

# 1. 初始化你的模型（与训练时结构一致）
model = YourModel(...)  # 填写你的模型参数
model.eval()  # 必须切换为评估模式

# 2. 生成与实际输入规格一致的模拟输入（示例为 (1,1,24)，可替换为你的输入维度）
dummy_input = torch.randn(1, 1, 24)  # 替换为你的输入形状（如 (1,3,224,224)）

# 3. 用 torch.jit.trace 追踪模型并保存（核心步骤）
traced_model = torch.jit.trace(model, dummy_input)
traced_model.save("traced_model.pt")  # 可自定义文件名，后续脚本中对应修改即可

print("模型已通过 torch.jit.trace 保存为 TorchScript 格式")
```

说明：保存时需确保模拟输入形状与你实际使用的输入维度一致，模型处于 eval 模式，否则会导致后续云端编译失败。

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
# 代理设置
# ======================
def setup_proxy(proxy_port: int = 10809) -> None:
    os.environ["http_proxy"] = f"http://127.0.0.1:{proxy_port}"
    os.environ["https_proxy"] = f"http://127.0.0.1:{proxy_port}"
    print(f"Proxy set to 127.0.0.1:{proxy_port}")

setup_proxy(10809)  # 替换为你的代理端口

# ======================
# 模型编译
# ======================
def load_and_compile_model(model_path: str, target_device: str, input_shape: tuple):
    """
    通用模型编译函数
    :param model_path: 本地 TorchScript 模型路径（.pt）
    :param target_device: 目标高通芯片（如 "SA8295P ADP"，从官网获取）
    :param input_shape: 模型输入形状（如 (1,1,24)，替换为你的输入维度）
    :return: 编译后的目标模型
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
            device=hub.Device(target_device, os="14"),  # os版本可根据官网调整
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
# 性能分析
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
# 云端推理
# ======================
def run_inference(model, input_shape: tuple):
   #读取真实数据集
    input_array = np.load("EDGE_1000.npz")["data"][0:1].reshape(1, 1, 24).astype(np.float32) 

    print(f"使用真实数据，shape: {input_array.shape}")
    

    print(f"Generated input data, shape: {input_array.shape}")
    
    try:
        print("Running inference...")
        inference_job = hub.submit_inference_job(
            model=model,
            device=hub.Device(target_device, os="14"),  # 自动适配编译时的目标芯片
            inputs=dict(image=[input_array]),
        )
        
        inference_job.wait()
        inference_out_dir = "inference_output"
        inference_job.download_results(inference_out_dir)
        print(f"Inference results saved to: {inference_out_dir}")
        
        # 递归查找 H5 文件中的 Dataset
        def get_h5_data(h5_item):
            if isinstance(h5_item, h5py.Dataset):
                return h5_item[:]
            for key in h5_item.keys():
                res = get_h5_data(h5_item[key])
                if res is not None:
                    return res
            return None
        
        # 读取 H5 结果文件
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
# 生成报告
# ======================
def generate_report(logits, chip_latency_ms: float, chip_mem_mb: float, p50_us: int, target_device: str):
    if logits is None:
        print("Failed to generate report: Inference results are empty")
        return
    
    print(f"             Qualcomm AI Hub 模型性能报告（Target Device: {target_device}）")
    print("=" * 65)
    print(f"【芯片侧硬件真实表现 (云端测量)】:")
    print(f" -> 纯硬件推理执行延迟: {chip_latency_ms:.4f} ms ({p50_us} us)")
    print(f" -> 芯片 NPU 峰值内存占用: {chip_mem_mb:.2f} MB")
# ======================
# 下载最终可部署的 DLC 模型
# ======================
def download_deployable_model(target_model: hub.Model, save_name: str = "final_deploy_model.dlc") -> None:
    """下载可部署到目标高通芯片的 DLC 模型文件（通用命名）"""
    try:
        target_model.download(save_name)
        print(f"\n[Success] Task completed. Deployable DLC model saved as: {save_name}")
    except Exception as e:
        print(f"\n[Failed] Model download failed: {str(e)}")

# ======================
# 主流程（可自由修改芯片、输入维度、模型路径）
# ======================
if __name__ == "__main__":
    # --------------------------
    # 可修改参数（根据自身需求调整）
    # --------------------------
    model_path = "SEResnet_model_traced_model.pt"  # 替换为你的 TorchScript 模型路径
    target_device = "SA8295P ADP"  # 替换为官网获取的目标芯片名称
    input_shape = (1, 1, 24)       # 替换为你的模型输入形状（示例：24维输入）
    
    # 执行主流程
    target_model = load_and_compile_model(model_path, target_device, input_shape)
    if not target_model:
        exit(1)

    chip_latency_ms, chip_mem_mb, p50_us = profile_model(target_model, target_device)
    logits = run_inference(target_model, input_shape)
    generate_report(logits, chip_latency_ms, chip_mem_mb, p50_us, target_device)
    download_deployable_model(target_model)

# 注：本脚本为通用版，可根据实际需求修改 "可修改参数" 部分，适配任意高通芯片与输入维度
# SA8295P ADP + 38维输入仅为示例，不局限于该配置
```

---

## 5. 运行命令

```bash
python qai_hub_deploy.py
```

## 6.性能分析
模型在目标芯片上编译完成后，Qualcomm AI Hub 官网会生成包含关键指标的性能分析报告。
- 推理延迟 (Inference Time)
- 内存占用 (Memory Usage)
- 计算单元 (Compute Units)

<img width="1278" height="529" alt="image" src="https://github.com/user-attachments/assets/5ffa0841-515a-4aad-955e-b12a67c7c7cf" />


---

## 7. 执行流程说明

1. 禁用本地 CUDA，避免环境干扰

2. 加载通过 torch.jit.trace 保存的 TorchScript 模型

3. 云端编译为目标芯片专用的 DLC 格式（可替换任意高通芯片）

4. 在真实处理器硬件上执行性能 profiling（延迟、内存测试）

5. 执行云端推理验证（适配任意输入维度，验证模型输出正常）

6. 生成通用版硬件性能报告（自动适配目标芯片）

7. 导出可直接部署到目标芯片的 `.dlc` 模型文件

---

## 8. 输出文件说明

|文件/目录|用途|
|---|---|
|`profile_output/`|硬件性能测试 JSON 结果（包含延迟、内存等详细数据）|
|`inference_output/`|云端推理 H5 输出文件（包含模型推理结果）|
|`final_deploy_model.dlc`|可直接部署到目标高通芯片的模型文件|
|`qai_hub_deploy.py`|完整部署脚本（可复用、可修改参数适配不同场景）|
|`traced_model.pt`|通过 torch.jit.trace 保存的 TorchScript 模型（需自行生成）|
---

## 9. 常见问题

### 9.1 连接超时 / 无法访问 Qualcomm AI Hub

- 代理端口错误：核对代理工具的端口（Clash=7890，V2Ray=10809），修改 `setup_proxy` 函数中的端口号

- 代理未开启：确保本地代理工具正常运行，可访问外网

- API Key 配置错误：重新生成 API Key，确保无多余空格、换行

### 9.2 模型编译失败

- 模型格式错误：未通过 `torch.jit.trace` 保存，重新执行模型保存代码

- 输入形状不匹配：模拟输入形状与模型实际输入维度不一致，修改模型保存和脚本中的输入形状

- 芯片名称错误：目标芯片名称与官网不一致，从官网重新获取并修改 `target_device` 参数

- 模型损坏：重新导出 TorchScript 模型，确保模型处于 eval 模式

### 9.3 性能分析 / 推理失败

- 输出目录权限不足：确保本地目录可读写，避免权限报错

- H5/JSON 文件未找到：等待任务完全执行完成，避免中途终止脚本

- os 版本不匹配：根据官网芯片要求，调整 `hub.Device` 中的 os 参数（默认 os="14"）

### 9.4 模型下载失败

- 网络不稳定：重新运行脚本，单独执行 `download_deployable_model` 函数

- 权限不足：修改 DLC 模型保存路径，确保目录可读写

---
## 10. 联系作者
- 如果你在使用本教程、运行脚本或配置 Qualcomm AI Hub 部署过程中遇到任何问题、需要协助或有建议，都可以直接联系我：
- 邮箱：ahufcy123@163.com
- 欢迎发送邮件交流技术问题、部署报错、模型适配方案，我会尽快回复。


---
