# FEDFT_code
FEDFT_code

# Local sequence collection example
python GRFG_with_nni.py --file-name openml_586_1 --episodes 512 --steps 5 

python GRFG_with_nni.py --file-name openml_586_2 --episodes 512 --steps 5 

python GRFG_with_nni.py --file-name openml_586_3 --episodes 512 --steps 5 

python GRFG_with_nni.py --file-name openml_586_4 --episodes 512 --steps 5 

# Server Side training and inference
python new_train_controller_KL.py --task_name openml_586 --train_top_k 2000 --top_k 50 --gpu 0 --batch_size 32 --epochs 60 --strategy_appendix 4-sigma-bias --keyword new_KL --new_gen 2000 --max_step_size 10 --align_top_k 1500


