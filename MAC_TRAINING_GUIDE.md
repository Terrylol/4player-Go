# 4人围棋 AI 训练指南 (Mac M1/M2/M3 版)

本指南专为 macOS 用户设计，利用 Apple Silicon (M系列芯片) 的 MPS 加速进行 AI 训练。

## 1. 环境准备

### 1.1 安装 Miniconda (Mac ARM64)
1. 下载 [Miniconda3 macOS Apple M1 64-bit pkg](https://docs.conda.io/en/latest/miniconda.html#macos-installers)。
2. 安装并打开终端。

### 1.2 创建虚拟环境
```bash
conda create -n go_ai python=3.10
conda activate go_ai
```

### 1.3 安装 PyTorch (MPS 支持)
PyTorch 对 Mac 的 MPS 加速支持已非常完善。
```bash
pip install torch torchvision torchaudio
```

### 1.4 安装其他依赖
进入项目 `training` 目录：
```bash
cd training
pip install -r requirements.txt
pip install onnx onnxruntime
```

## 2. 运行测试训练

我们建议先运行一次快速训练，确保环境正常。

```bash
# 训练 9x9 棋盘，运行 2 轮，每轮 5 局
python train.py
```
*注意：您可以在 `train.py` 文件底部修改 `train(board_size=9, epochs=2, games_per_epoch=5)` 参数进行测试。*

如果看到输出 `Using device: mps`，说明已成功调用 GPU 加速。

## 3. 正式训练

确认测试通过后，您可以增加训练量以提升 AI 水平。

建议配置：
- `epochs=50` (轮数)
- `games_per_epoch=50` (每轮对局数)

```python
# 修改 train.py
if __name__ == "__main__":
    train(board_size=9, epochs=50, games_per_epoch=50)
```

## 4. 导出与部署

训练完成后，导出模型：
```bash
python export_onnx.py
```

将生成的 `model.onnx` 重命名为 `model_9x9.onnx`，并移动到前端目录：
```bash
mv model.onnx ../public/models/model_9x9.onnx
```

现在启动前端游戏 `npm run dev`，切换到 9x9 棋盘即可与新训练的 AI 对战。
