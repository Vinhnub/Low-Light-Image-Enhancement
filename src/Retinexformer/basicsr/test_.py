import torch
from pprint import pprint


pth_path = r"E:/PythonFile/Project/Low-Light-Image-Enhancement/src/Retinexformer/experiments/LoL_v1/base/model_weight/best_psnr_23.63_96000.pth"

ckpt = torch.load(pth_path, map_location="cpu")

print("=" * 80)
print("CHECKPOINT TYPE")
print("=" * 80)
print(type(ckpt))

print("\n" + "=" * 80)
print("TOP LEVEL KEYS")
print("=" * 80)

if isinstance(ckpt, dict):
    pprint(list(ckpt.keys()))
else:
    print("Checkpoint is not dict")
    exit()


# ------------------------------------------------
# find actual state dict
# ------------------------------------------------
candidate_keys = [
    "params",
    "params_ema",
    "state_dict",
    "model",
    "net",
]

state_dict = None
used_key = None

for k in candidate_keys:
    if k in ckpt:
        state_dict = ckpt[k]
        used_key = k
        break

if state_dict is None:
    state_dict = ckpt
    used_key = "root"

print("\n" + "=" * 80)
print("USING STATE_DICT KEY")
print("=" * 80)
print(used_key)

print("\n" + "=" * 80)
print("NUMBER OF PARAMETERS")
print("=" * 80)
print(len(state_dict))


print("\n" + "=" * 80)
print("FIRST 100 PARAMETER KEYS")
print("=" * 80)

for i, (k, v) in enumerate(state_dict.items()):
    print(f"{i:03d} | {k} | {tuple(v.shape)}")

    if i >= 99:
        break


# ------------------------------------------------
# infer architecture hints
# ------------------------------------------------
print("\n" + "=" * 80)
print("ARCHITECTURE HINTS")
print("=" * 80)

for k, v in state_dict.items():

    if "conv1.weight" in k:
        print(f"Possible n_feat from {k}: {v.shape[0]}")

    if "body" in k:
        print("Found body module:", k)
        break

# largest tensors
print("\n" + "=" * 80)
print("LARGEST TENSORS")
print("=" * 80)

sorted_items = sorted(
    state_dict.items(),
    key=lambda x: x[1].numel(),
    reverse=True
)

for k, v in sorted_items[:20]:
    print(
        f"{k:<70} "
        f"shape={tuple(v.shape)} "
        f"params={v.numel():,}"
    )