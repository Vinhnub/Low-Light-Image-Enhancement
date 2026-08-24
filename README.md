# Low-Light-Image-Enhancement

Thư mục làm việc: DE190323_Vinh
Môi trường conda: DE190323_Vinh_Mamba, conda activate DE190323_Vinh_Mamba (dùng để kích hoạt môi trường)
Đầu tiên vào thư mục làm việc xóa toàn bộ thư mục Low-Light-Image-Enhancement
Sau đó clone lại code: git clone https://github.com/Vinhnub/Low-Light-Image-Enhancement.git
Tạo một thư mục results trong HVI-CIDNet, sau đó lại tạo một thư mục traning trong results
Đưa đường dẫn dataset vào trong option
Lệnh để train model: CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 train_ddp.py --dataset lolv2_real (ở đây có thể tùy chỉnh giữa các dataset, lol_v1, lolv2_syn)