# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import io
import os
import time
from collections import defaultdict, deque
import datetime
from contextlib import contextmanager

import torch
import torch.distributed as dist


@contextmanager
def timer(name):
    """Context manager for timing code execution."""
    start = time.time()
    yield
    end = time.time()
    #print(f"[GPU {get_gpu()}] {name}: {end-start:.3f}s")


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save({'state_dict_ema':checkpoint}, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def get_gpu():
    if not is_dist_avail_and_initialized():
        return 0
    return int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else get_rank() % torch.cuda.device_count()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'SLURM_PROCID' in os.environ:
        if args.debug:
            print('Using SLURM')
        args.rank = int(os.environ['SLURM_PROCID'])
        # IMPORTANT: Always set GPU index based on LOCAL_RANK for proper distribution
        if 'LOCAL_RANK' in os.environ:
            if args.debug:
                print(f'Using LOCAL_RANK {os.environ["LOCAL_RANK"]}')
            args.gpu = int(os.environ['LOCAL_RANK'])
        else:
            if args.debug:
                print('Using SLURM_PROCID for GPU assignment')
            args.gpu = args.rank % torch.cuda.device_count()

        args.world_size = int(os.environ['SLURM_NTASKS'])
        
        if args.debug:
            print(f"Process rank: {args.rank}, assigned to GPU: {args.gpu}")
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        if args.debug:
            print('Using environment variables for distributed configuration')
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        
        # IMPORTANT: Always set GPU index based on LOCAL_RANK for proper distribution
        args.gpu = int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else args.rank % torch.cuda.device_count()
        
        if args.debug:
            print(f"Process rank: {args.rank}, world size: {args.world_size}, using GPU: {args.gpu}")
    else:
        if args.debug:
            print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True
    
    # Print available GPUs to help diagnose issues
    if args.debug:
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Ensure CUDA is available
    if not torch.cuda.is_available():
        print("Warning: CUDA not available but trying to use distributed mode")
        return
    
    # Set the CUDA device
    try:
        torch.cuda.set_device(args.gpu)
        if args.debug:
            print(f"Process {args.rank}: Set CUDA device to GPU {args.gpu}")
            # Verify we're on the right device
            current_device = torch.cuda.current_device()
            if current_device != args.gpu:
                print(f"Warning: Expected device {args.gpu}, got {current_device}")
            
            # Print memory info for this GPU
            print(f"GPU {args.gpu} memory allocated: {torch.cuda.memory_allocated(args.gpu) / 1e9:.3f} GB")
    except Exception as e:
        print(f"Error setting CUDA device: {e}")

    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)

    try:
        if args.debug:
            print(f"Initializing process group with backend {args.dist_backend}")
            print(f"Using init method: {args.dist_url}")
            print(f"World size: {args.world_size}, rank: {args.rank}")
        torch.distributed.init_process_group(backend=args.dist_backend, 
                                          init_method=args.dist_url,
                                          world_size=args.world_size, 
                                          rank=args.rank)
        if args.debug:
            print(f"Process group initialized successfully")
    except Exception as e:
        print(f"Failed to initialize distributed process group: {e}")
        args.distributed = False
        return

    try:
        if args.debug:
            print(f"|-Barrier for process {args.rank} on GPU {args.gpu}")
        
        # Add these two lines before the barrier for debugging
        dist.all_reduce(torch.ones(1).to(args.gpu))
        print(f"| Process {args.rank} passed all_reduce test", flush=True)
        
        # Replace standard barrier with timeout version
        torch.distributed.barrier()
        print(f'| Process {args.rank} passed barrier on GPU {args.gpu}', flush=True)
    except Exception as e:
        print(f"Error during barrier: {e}")
        # Continue execution even if barrier fails
        pass
    # Uncomment if needed - disable printing for non-master processes
    setup_for_distributed(args.rank == 0) 

    if args.debug:
        print(f'| Process {args.rank} done with init process group on GPU {args.gpu}', flush=True)


# if 'pos_embed' in state_dict:
def interpolate_pos_embed(model, state_dict):
    pos_embed_checkpoint = state_dict['pos_embed']
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.patch_embed.num_patches
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches
    # import ipdb; ipdb.set_trace()
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)
    # class_token and dist_token are kept unchanged
    if orig_size != new_size:
        print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        state_dict['pos_embed'] = new_pos_embed