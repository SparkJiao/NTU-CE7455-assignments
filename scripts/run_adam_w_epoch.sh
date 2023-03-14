conf=$1
conf_name=$2
num_rank=$3
gpu=$4
port=$5

count=1
for arg in "$@"; do
  if [ "$count" -gt "5" ]; then
    if [ "$num_rank" -gt "1" ]; then
      echo "CUDA_VISIBLE_DEVICES=$gpu python -m torch.distributed.run --nproc_per_node $num_rank --master_port $port trainer_torch_fsdp_wandb.py -cp $conf -cn $conf_name num_train_epochs=${arg}"

      CUDA_VISIBLE_DEVICES=$gpu python -m torch.distributed.run --nproc_per_node $num_rank --master_port $port trainer_torch_fsdp_wandb.py -cp $conf -cn $conf_name num_train_epochs=${arg}
    else
      echo "CUDA_VISIBLE_DEVICES=$gpu python trainer_torch_fsdp_wandb.py -cp $conf -cn $conf_name num_train_epochs=${arg}"

      CUDA_VISIBLE_DEVICES=$gpu python trainer_torch_fsdp_wandb.py -cp $conf -cn $conf_name num_train_epochs=${arg}
    fi
  fi
  let count=count+1
done;