# test_ddp.py
import os, torch, torch.distributed as dist

def main():
    rank        = int(os.environ['RANK'])
    world_size  = int(os.environ['WORLD_SIZE'])
    local_rank  = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    print(f"[{rank}] initializing…", flush=True)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
    )
    print(f"[{rank}] init complete, doing all_reduce…", flush=True)

    # allocate ON the GPU
    x = torch.ones(1, device=f"cuda:{local_rank}")
    dist.all_reduce(x)
    print(f"[{rank}] all_reduce OK → {x.item()}", flush=True)

if __name__ == '__main__':
    main()
