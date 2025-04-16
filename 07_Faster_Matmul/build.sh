EXEC="sgemm"
rm -rf $EXEC > /dev/null
nvcc main.cu ../00_Utils/cudaUtils.cu -I./../00_Utils -o $EXEC -arch=compute_50