# get param for compute, if it is a number (i.e., in expected format), use it!
IFS='.' read -r X Y <<< "$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader)"

EXEC="sgemm"
echo "Building $EXEC..."

rm -rf $EXEC > /dev/null
echo "Removing old executable from $PWD"

nvcc main.cu ../00_Utils/cudaUtils.cu -I./../00_Utils -o $EXEC -arch=compute_$X$Y
echo "Build successful, command used := nvcc main.cu ../00_Utils/cudaUtils.cu -I./../00_Utils -o $EXEC -arch=compute_$X$Y"