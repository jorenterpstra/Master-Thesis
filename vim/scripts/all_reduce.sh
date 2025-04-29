export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
python -m torch.distributed.run \
    --nproc_per_node=2 \
    test_ddp.py