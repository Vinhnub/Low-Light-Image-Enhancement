import cv2
import numpy as np
import argparse
import os

def interleave_images(img1_path, img2_path, patch_size, output_path):
    # Load images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None:
        raise ValueError(f"Could not read image 1 at {img1_path}")
    if img2 is None:
        raise ValueError(f"Could not read image 2 at {img2_path}")
        
    # Ensure they have the same shape, if not resize img2 to img1 shape
    if img1.shape != img2.shape:
        print(f"Warning: image sizes do not match. Resizing image 2 to match image 1.")
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
    h, w, c = img1.shape
    
    # Calculate padding if dimensions are not divisible by patch_size
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    
    if pad_h > 0 or pad_w > 0:
        print(f"Padding image by {pad_h} in height and {pad_w} in width to match patch size {patch_size}.")
        # Use edge padding or reflect padding
        img1 = cv2.copyMakeBorder(img1, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        img2 = cv2.copyMakeBorder(img2, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        
    padded_h, padded_w, _ = img1.shape
    
    # Number of patches in height and width
    H_p = padded_h // patch_size
    W_p = padded_w // patch_size
    
    # 1. Trải phẳng (flatten) các patch của từng ảnh
    # Chuyển ảnh thành ma trận các patch: (H_p, W_p, patch_size, patch_size, c)
    patches1 = img1.reshape(H_p, patch_size, W_p, patch_size, c).swapaxes(1, 2)
    patches2 = img2.reshape(H_p, patch_size, W_p, patch_size, c).swapaxes(1, 2)
    
    # Trải phẳng thành 1D (danh sách các patch): (H_p * W_p, patch_size, patch_size, c)
    patches1_flat = patches1.reshape(-1, patch_size, patch_size, c)
    patches2_flat = patches2.reshape(-1, patch_size, patch_size, c)
    
    # 2. Xếp xen kẽ (interleave) 2 danh sách patch
    # Vì output phải CÙNG SIZE với input, tổng số patch trong ảnh kết quả chỉ là H_p * W_p.
    # Do đó, ta sẽ xen kẽ: Patch 1 ảnh 1, Patch 1 ảnh 2, Patch 2 ảnh 1, Patch 2 ảnh 2...
    # và dừng lại khi mảng chứa đủ H_p * W_p patches.
    num_patches = H_p * W_p
    interleaved_flat = np.empty((num_patches, patch_size, patch_size, c), dtype=img1.dtype)
    
    # Số lượng patch mỗi ảnh đóng góp là num_patches // 2
    half_patches = num_patches // 2
    interleaved_flat[0:half_patches*2:2] = patches1_flat[:half_patches]
    interleaved_flat[1:half_patches*2:2] = patches2_flat[:half_patches]
    
    # Nếu tổng số patch là số lẻ, ta cần thêm 1 patch cuối của ảnh 1
    if num_patches % 2 != 0:
        interleaved_flat[-1] = patches1_flat[half_patches]
    
    # 3. Đưa lại thành 2D
    # Vì cùng size với input nên ta dùng lại đúng số hàng (H_p) và số cột (W_p)
    interleaved_grid = interleaved_flat.reshape(H_p, W_p, patch_size, patch_size, c)
    
    # Ghép các patch lại thành ảnh hoàn chỉnh
    output_img = interleaved_grid.swapaxes(1, 2).reshape(H_p * patch_size, W_p * patch_size, c)
                
    # Lấy đúng kích thước gốc nếu đã padding (cắt bỏ phần padding nếu bạn muốn ảnh xuất ra giống hệt input 100%)
    # Ở đây tôi giữ lại phần đã padding để đảm bảo không mất patch nào.
    output_img = output_img[:padded_h, :padded_w]
    
    # Save output
    cv2.imwrite(output_path, output_img)
    print(f"Successfully saved interleaved image to {output_path}")
    if pad_h > 0 or pad_w > 0:
        print(f"Note: Input images were padded. Original size: ({h}, {w}), Padded size: ({padded_h}, {padded_w})")
    print(f"Final output size: ({padded_h}, {padded_w})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Interleave patches of two images.")
    parser.add_argument('--img1', type=str, required=True, help="Path to first image")
    parser.add_argument('--img2', type=str, required=True, help="Path to second image")
    parser.add_argument('--patch_size', type=int, default=64, help="Patch size for interleaving")
    parser.add_argument('--output', type=str, default='output/interleaved_output.png', help="Output path")
    
    args = parser.parse_args()
    
    interleave_images(args.img1, args.img2, args.patch_size, args.output)
