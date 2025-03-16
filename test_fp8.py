import torch


mat1 = torch.randn(2, 3)
mat2 = torch.randn(3, 3)
res = torch.mm(mat1, mat2)
print(f"mat1.dtype: {mat1.dtype}, mat2.dtype: {mat2.dtype}, res.dtype: {res.dtype}")
print(f"mat1 @ mat2 = {res}")

mat1_bf16 = mat1.to(torch.bfloat16)
mat2_bf16 = mat2.to(torch.bfloat16)
res_bf16 = torch.mm(mat1_bf16, mat2_bf16)
print(f"mat1_bf16.dtype: {mat1_bf16.dtype}, mat2_bf16.dtype: {mat2_bf16.dtype}, res_bf16.dtype: {res_bf16.dtype}")
print(f"mat1_bf16 @ mat2_bf16 = {res_bf16}")

mat1_fp8_e4m3 = mat1.to(torch.float8_e4m3fn)
mat2_fp8_e4m3 = mat2.to(torch.float8_e4m3fn)
res_fp8_e4m3 = torch.mm(mat1_fp8_e4m3, mat2_fp8_e4m3)
print(f"mat1_fp8_e4m3.dtype: {mat1_fp8_e4m3.dtype}, mat2_fp8_e4m3.dtype: {mat2_fp8_e4m3.dtype}, res_fp8_e4m3.dtype: {res_fp8_e4m3.dtype}")
print(f"mat1_fp8_e4m3 @ mat2_fp8_e4m3 = {res_fp8_e4m3}")

print(f"fp8_e4m3 max: {torch.finfo(torch.float8_e4m3fn).max}, min: {torch.finfo(torch.float8_e4m3fn).min}")

