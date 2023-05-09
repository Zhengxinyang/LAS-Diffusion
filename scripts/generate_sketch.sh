# sketch-conditioned generation
# please give a rough view information of the object in the sketch

## creative design
python generate.py --model_path ./checkpoints/shape_sketch/epoch\=299.ckpt --generate_method generate_based_on_sketch  --num_generate 4 --steps 50 --image_path ./demo_data/00.png --view_information 3
python generate_super_resolution.py --model_path ./checkpoints/shape_super/epoch\=499.ckpt --generate_method generate_meshes --npy_path ./outputs/shape_sketch_epoch=299.ckpt_True_demo_data_00_1.0_3 --save_npy True --save_mesh True --level 0.0 --steps 20


python generate.py --model_path ./checkpoints/shape_sketch/epoch\=299.ckpt --generate_method generate_based_on_sketch  --num_generate 4 --steps 50 --image_path ./demo_data/01.png --view_information 1
python generate_super_resolution.py --model_path ./checkpoints/shape_super/epoch\=499.ckpt --generate_method generate_meshes --npy_path ./outputs/shape_sketch_epoch=299.ckpt_True_demo_data_01_1.0_1 --save_npy True --save_mesh True --level 0.0 --steps 20

## add bars
python generate.py --model_path ./checkpoints/shape_sketch/epoch\=299.ckpt --generate_method generate_based_on_sketch  --num_generate 4 --steps 50 --image_path ./demo_data/02.png --view_information 1
python generate_super_resolution.py --model_path ./checkpoints/shape_super/epoch\=499.ckpt --generate_method generate_meshes --npy_path ./outputs/shape_sketch_epoch=299.ckpt_True_demo_data_02_1.0_1 --save_npy True --save_mesh True --level 0.0 --steps 20


python generate.py --model_path ./checkpoints/shape_sketch/epoch\=299.ckpt --generate_method generate_based_on_sketch  --num_generate 4 --steps 50 --image_path ./demo_data/03.png --view_information 1
python generate_super_resolution.py --model_path ./checkpoints/shape_super/epoch\=499.ckpt --generate_method generate_meshes --npy_path ./outputs/shape_sketch_epoch=299.ckpt_True_demo_data_03_1.0_1 --save_npy True --save_mesh True --level 0.0 --steps 20