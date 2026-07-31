import os
low_dir = "E:/PythonFile/Project/Low-Light-Image-Enhancement/mydata/dataset/dataset/LOLv2-real/Test/Input"
normal_dir = "E:/PythonFile/Project/Low-Light-Image-Enhancement/mydata/dataset/dataset/LOLv2-real/Test/GT"
output_txt = "E:/PythonFile/Project/Low-Light-Image-Enhancement/src/LREMNet/data/LOLv2_val.txt"
with open(output_txt, "w") as f:
    for img_name in sorted(os.listdir(low_dir)):
        if img_name.endswith('.png'):
            low_path = os.path.join(low_dir, img_name)
            normal_path = os.path.join(normal_dir, img_name)
            f.write(f"{low_path} {normal_path}\n")
print("Tạo file TXT thành công!")