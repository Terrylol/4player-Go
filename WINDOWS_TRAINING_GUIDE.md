# 4人围棋 AI 训练与部署指南 (Windows 版)

本指南将指导您在 Windows 环境下，使用 NVIDIA 显卡训练一个 9x9 棋盘的 4 人围棋 AI，并将其部署到游戏中。

## 1. 环境准备

### 1.1 安装 Miniconda
推荐使用 Miniconda 来管理 Python 环境。
1. 下载并安装 [Miniconda for Windows](https://docs.conda.io/en/latest/miniconda.html)。
2. 安装完成后，打开 **Anaconda Powershell Prompt**。

### 1.2 创建虚拟环境
在命令行中执行以下命令：
```powershell
conda create -n go_ai python=3.10
conda activate go_ai
```

### 1.3 安装 PyTorch (带 CUDA 支持)
请根据您的显卡驱动版本选择合适的命令。通常情况下（CUDA 11.8）：
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
*如果不确定 CUDA 版本，可以运行 `nvidia-smi` 查看。如果需要其他版本，请访问 [PyTorch 官网](https://pytorch.org/get-started/locally/) 获取安装命令。*

### 1.4 安装其他依赖
进入项目目录下的 `training` 文件夹，安装剩余依赖：
```powershell
cd training
pip install -r requirements.txt
pip install onnx onnxruntime
```

## 2. 训练 AI

我们提供了现成的训练脚本 `train.py`。默认配置下，它会训练一个 9x9 的 AI。

### 2.1 开始训练
在 `training` 目录下运行：
```powershell
python train.py
```

### 2.2 自定义参数
您可以编辑 `train.py` 文件的底部来修改训练参数：
```python
if __name__ == "__main__":
    # board_size: 棋盘大小 (9, 13, 19)
    # epochs: 训练轮数 (建议 50-100)
    # games_per_epoch: 每轮自我对弈局数 (建议 50+)
    train(board_size=9, epochs=50, games_per_epoch=100)
```
*注意：训练时间取决于您的显卡性能。对于“正常对弈”水平，9x9 棋盘建议训练至少 50 轮。*

训练过程中，程序会自动保存检查点 `checkpoint_X.pth` 和最新模型 `latest_model.pth`。

## 3. 导出模型

训练完成后，需要将 PyTorch 模型转换为 ONNX 格式，以便在网页前端运行。

运行以下命令：
```powershell
python export_onnx.py
```
这会读取 `latest_model.pth` 并生成 `model.onnx` 文件。

*如果您训练的是非 9x9 棋盘，请修改 `export_onnx.py` 中的 `board_size` 参数。*

## 4. 部署到游戏

### 4.1 放置模型文件
将生成的 `model.onnx` 文件重命名并移动到前端的 `public/models/` 目录下：

1. 在项目根目录的 `public` 文件夹下创建 `models` 文件夹（如果不存在）。
2. 将 `training/model.onnx` 复制到 `public/models/model_9x9.onnx`。

*如果您训练了其他尺寸，请命名为 `model_13x13.onnx` 或 `model_19x19.onnx`。*

### 4.2 运行游戏
现在启动前端：
```powershell
npm run dev
```
打开浏览器，选择 **9x9** 棋盘，并在游戏设置中选择 **PvE** 或 **EvE** 模式。游戏会自动加载您刚刚训练的 AI 模型。如果加载失败（例如文件未找到），游戏会自动降级使用内置的简单 AI。

## 常见问题

**Q: 训练太慢怎么办？**
A: `train.py` 目前使用的是单进程自我对弈以确保 Windows 兼容性。如果您熟悉 Python，可以尝试修改代码使用多进程（需注意 Windows 下 `spawn` 启动方式的限制）。或者减少 `games_per_epoch`。

**Q: 显存不足 (OOM)？**
A: 减小 `games_per_epoch` 不会影响显存，显存主要由 `batch_size` 决定（目前代码中未显式使用 batch，是一局一局跑的）。如果遇到问题，请检查是否有其他程序占用显存。

**Q: 可以在 CPU 上训练吗？**
A: 可以，代码会自动检测。如果未检测到 CUDA，将使用 CPU。但这会非常慢，仅建议用于调试。
