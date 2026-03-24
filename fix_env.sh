echo "🔍 诊断当前 PyTorch CUDA 版本..."
TORCH_CUDA=$(python -c "import torch; print(torch.version.cuda)")
echo "PyTorch 期望的 CUDA 版本: $TORCH_CUDA"

echo "📂 搜索 /usr/local 中的 CUDA 安装目录..."
CUDA_PATHS=$(find /usr/local -maxdepth 1 -type d -name "cuda-*" 2>/dev/null)

MATCH_PATH=""
for p in $CUDA_PATHS; do
    if [[ "$p" == *"$TORCH_CUDA"* ]]; then
        MATCH_PATH=$p
        break
    fi
done

if [ -z "$MATCH_PATH" ]; then
    echo "❌ 未找到匹配的 CUDA $TORCH_CUDA，请手动指定 CUDA_HOME"
    exit 1
else
    echo "✅ 找到匹配 CUDA 路径: $MATCH_PATH"
fi

echo ""
echo "------------------------------------------------"
echo "请执行以下命令（或将其加入 ~/.bashrc）:"
echo "------------------------------------------------"
echo "export CUDA_HOME=$MATCH_PATH"
echo "export PATH=\$CUDA_HOME/bin:\$PATH"
echo "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH"
echo "export TORCH_CUDA_ARCH_LIST=\"8.9\"   # RTX 4090 架构"
echo "------------------------------------------------"
