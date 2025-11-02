import time

import mlx.core as mx
import torch

import GPT_SoVITS.text.g2pw.converter
from GPT_SoVITS.Accel.logger import timer
from GPT_SoVITS.text.g2pw.converter import G2PWConverter
from GPT_SoVITS.text.g2pw.onnx_api import G2PWOnnxConverter

a = G2PWOnnxConverter(model_dir="GPT_SoVITS/text/G2PWModel", enable_non_tradional_chinese=True)
b = G2PWConverter(model_source="GPT_SoVITS/pretrained_models/g2pw-chinese")

GPT_SoVITS.text.g2pw.converter.device = torch.device("cpu")
GPT_SoVITS.text.g2pw.converter.dtype = torch.float32

res_a = a("我在我青春韶华的时候遇到了你")
res_b = b("我在我青春韶华的时候遇到了你")

timer.clear()

print(res_a)

print(res_b)

start_time = time.perf_counter()
for i in range(200):
    test_torch = a("我在我青春韶华的时候遇到了你")
end_time = time.perf_counter()
print(f"ONNX Inference Time: {end_time - start_time}s")

start_time = time.perf_counter()
for i in range(200):
    test_mlx = b("我在我青春韶华的时候遇到了你")
    mx.eval(test_mlx)

end_time = time.perf_counter()
print(f"MLX Inference Time: {(end_time - start_time)}s")

timer.summary()


# start_time = time.perf_counter()
# for i in range(100):
#     test_torch = c(input_ids, phoneme_masks, char_ids, position_ids)

# end_time = time.perf_counter()
# print(f"Torch Inference Time: {end_time - start_time}s")

# start_time = time.perf_counter()
# for i in range(100):
#     test_mlx = d(input_ids_mlx, phoneme_masks_mlx, char_ids_mlx, position_ids_mlx)
#     mx.eval(test_mlx)

# end_time = time.perf_counter()
# print(f"MLX Inference Time: {end_time - start_time}s")

# timer.clear()

# start_time = time.perf_counter()
# for i in range(100):
#     test_torch = c(input_ids, phoneme_masks, char_ids, position_ids)

# end_time = time.perf_counter()
# print(f"Torch Inference Time: {end_time - start_time}s")

# start_time = time.perf_counter()
# for i in range(100):
#     test_mlx = d(input_ids_mlx, phoneme_masks_mlx, char_ids_mlx, position_ids_mlx)
#     mx.eval(test_mlx)

# end_time = time.perf_counter()
# print(f"MLX Inference Time: {end_time - start_time}s")
