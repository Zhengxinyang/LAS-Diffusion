# please modify the sdf_folder and sketch_folder to your own path

# occpuancy-diffuion module
python train.py --data_class class_5 --name shape_sketch --batch_size 16 --new True --continue_training False --image_size 64 --training_epoch 300 --ema_rate 0.995 --base_channels 32 --noise_schedule linear --save_last False --save_every_epoch 50 --with_attention True --use_text_condition False --use_sketch_condition True --kernel_size 4.0  --view_information_ratio 2.0  --lr 2e-4 --optimizier adam --data_augmentation True --sdf_folder /home/D/dataset/shapenet_sdf --sketch_folder /home/D/dataset/shapenet_edge_our_new

# SDF-diffuion module
python train_super_resolution.py --data_class class_5 --name shape_super --batch_size 4 --new True --continue_training False --training_epoch 500  --split_dataset True --sdf_folder /home/D/dataset/shapenet_sdf 


