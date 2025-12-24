run_name="URM-arcagi1"
checkpoint_path="checkpoints/${run_name}" 
mkdir -p $checkpoint_path

torchrun --nproc-per-node 8 pretrain.py \
data_path=data/arc1concept-aug-1000 \
arch=urm arch.loops=16 arch.H_cycles=2 arch.L_cycles=6 arch.num_layers=4 \
epochs=200000 \
eval_interval=2000 \
puzzle_emb_lr=1e-2 \
weight_decay=0.1 \
+run_name=$run_name \
+checkpoint_path=$checkpoint_path \
+ema=True
