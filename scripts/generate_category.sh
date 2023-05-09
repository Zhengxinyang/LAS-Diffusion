# unconditional generation
# replace airplane with chair, car, rifle, table etc. to generate other categories

python generate.py --model_path ./checkpoints/airplane/epoch\=3999.ckpt --generate_method generate_unconditional --num_generate 16 --steps 50
python generate_super_resolution.py --model_path ./checkpoints/shape_super/epoch\=499.ckpt --generate_method generate_meshes --npy_path ./outputs/airplane_epoch\=3999.ckpt_True_50_0.0_unconditional --save_npy True --save_mesh True --level 0.0 --steps 20

# category-conditional generation

python generate.py --model_path ./checkpoints/shape_five/epoch\=3999.ckpt --generate_method generate_based_on_class --data_class chair --num_generate 16 --steps 50
python generate_super_resolution.py --model_path ./checkpoints/shape_super/epoch\=499.ckpt --generate_method generate_meshes --npy_path ./outputs/shape_five_epoch\=3999.ckpt_True_50_0.0_1.0_chair --save_npy True --save_mesh True --level 0.0 --steps 20