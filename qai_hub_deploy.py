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
    :param input_shape: 模型输入形状（如 (1,1,38)，替换为你的输入维度）
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
    # 生成与实际输入规格一致的模拟输入（替换为你的输入维度）
    input_array = np.load("EDGE_1000.npz")["data"][0:1].reshape(1, 1, 24).astype(np.float32)
    print(f"Generated input data, shape: {input_array.shape}")
    
    try:
        print("Running inference...")
        inference_job = hub.submit_inference_job(
            model=model,
            device=hub.Device(target_device, os="14"), # 自动适配编译时的目标芯片
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
    
    print("\n" + "=" * 65)
    print(f"             Qualcomm AI Hub 模型性能报告（Target Device: {target_device}）")
    print("=" * 65)
    print(f"【芯片侧硬件真实表现 (云端测量)】:")
    print(f" -> 纯硬件推理执行延迟: {chip_latency_ms:.4f} ms ({p50_us} us)")
    print(f" -> 芯片 NPU 峰值内存占用: {chip_mem_mb:.2f} MB")
    print("-" * 65)



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
    input_shape = (1, 1, 24)       # 替换为你的模型输入形状（示例：38维输入）
    
    # 执行主流程
    target_model = load_and_compile_model(model_path, target_device, input_shape)
    if not target_model:
        exit(1)

    chip_latency_ms, chip_mem_mb, p50_us = profile_model(target_model, target_device)
    logits = run_inference(target_model, input_shape)
    generate_report(logits, chip_latency_ms, chip_mem_mb, p50_us, target_device)
    download_deployable_model(target_model)
