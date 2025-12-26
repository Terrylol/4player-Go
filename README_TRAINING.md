# 4人围棋 AI 训练与部署指南

## 1. 训练环境搭建 (Python)
此代码需要在具备 GPU 的环境中运行（推荐 NVIDIA显卡 + CUDA）。

### 安装依赖
```bash
pip install -r training/requirements.txt
```

### 文件说明
*   `training/go_env.py`: 4人围棋游戏逻辑环境 (Gym-like API)。
*   `training/network.py`: PyTorch 定义的 ResNet 神经网络结构。
*   `training/mcts.py`: 蒙特卡洛树搜索算法。
*   `training/train.py`: 主训练脚本 (Self-Play -> Train Loop)。
*   `training/export_onnx.py`: 模型导出脚本。

## 2. 开始训练
运行以下命令开始训练：
```bash
python training/train.py
```
训练过程中会生成 `checkpoint_N.pth` 文件。

## 3. 导出模型
当训练达到满意效果后，将 PyTorch 模型导出为 ONNX 格式，以便在浏览器中使用：
```bash
python training/export_onnx.py
```
这将生成 `model.onnx` 文件。

## 4. 前端部署策略 (Deployment Strategy)

### 步骤 A: 放置模型
将生成的 `model.onnx` 文件复制到前端项目的 `public/` 目录下：
```bash
cp model.onnx public/model.onnx
```

### 步骤 B: 安装前端推理库
在前端项目中安装 ONNX Runtime Web：
```bash
npm install onnxruntime-web
```

### 步骤 C: 前端调用 (伪代码)
我们已经在 `src/game/NeuralAI.ts` (需创建) 中准备好了调用代码。
AI 会加载 `model.onnx`，将当前棋盘状态转换为 Tensor 输入，获取 Policy 输出，然后选择概率最高的合法落子点。

```typescript
import * as ort from 'onnxruntime-web';

async function runInference(boardState) {
    const session = await ort.InferenceSession.create('/model.onnx');
    const input = prepareTensor(boardState);
    const results = await session.run({ input: input });
    return getBestMove(results.policy);
}
```
