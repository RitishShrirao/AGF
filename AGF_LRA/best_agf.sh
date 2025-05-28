# listops
python3 run_tasks.py --model svd --task listops --K 5 --poly_type jacobi --alpha 2.0 --beta "-1.0" --fixI True --reg 0.01 

# image
python3 run_tasks.py --model svd --task image --K 6 --poly_type jacobi --alpha 1.5 --beta 2.0 --fixI True --reg 0.1 

# pathfinder
python3 run_tasks.py --model svd --task pathfinder32-curv_contour_length_14 --poly_type jacobi --K 4 --alpha 1.5 --beta "-1.0" --fixI --reg 0.0001 

# rerieval
python3 run_tasks.py --model svd --task retrieval  --poly_type jacobi --K 5 --alpha 2.0 --beta "-1.5" --fixI True --reg 0.0001 

# text
python3 run_tasks.py --model svd --task text --poly_type jacobi --K 5 --alpha 2.0 --beta "-2.0" --fixI True --reg 0.01 
