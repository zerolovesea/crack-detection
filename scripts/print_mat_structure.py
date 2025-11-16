import scipy.io as sio
import os

mat_dir = "CrackForest-dataset/groundTruth"

# 随便选一个 .mat 文件（比如第一个）
for f in os.listdir(mat_dir):
    if f.endswith(".mat"):
        sample = os.path.join(mat_dir, f)
        break

print("检查文件：", sample)
mat = sio.loadmat(sample)

# 查看顶层 key
print("\n== 顶层 keys ==")
for k in mat.keys():
    print(" ", k, type(mat[k]))

gt = mat["groundTruth"]
print("\n== groundTruth 内容 ==")
print("type:", type(gt))
print("shape:", getattr(gt, "shape", None))
print(gt)