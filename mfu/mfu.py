import csv
import os

from hardware.gpu import gpu_map


def get_attn_decode_mfu(config, target_bs, kv_len, device_type, use_fp8_kv, tp_size):
    gpu = gpu_map[device_type]
    tp_num_heads = config.num_attention_heads // tp_size
    if config.attn_type == "MHA/GQA":
        tp_num_kv_heads = config.num_key_value_heads // tp_size
        head_dim = config.head_dim
        file_name = f"bench_data/mha/decode/{device_type.lower()}/{tp_num_heads}-{tp_num_kv_heads}-{head_dim}.csv"
    elif config.attn_type == "MLA":
        head_dim = f"{config.kv_lora_rank}-{config.qk_rope_head_dim}"
        file_name = f"bench_data/mla/decode/{device_type.lower()}/{tp_num_heads}-{head_dim}.csv"
    if not os.path.exists(file_name):
        print(f"Warning: {file_name} not exists")
        return gpu.mfu

    # row: dtype,kv_dtype,batch_size,kv_len,latency,mfu
    kv_dtype = "fp8" if use_fp8_kv else "bf16"
    rows = list()
    with open(file_name, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row[1] != kv_dtype:
                continue
            rows.append(row)

    # Find the row with closest batch_size and kv_len to target values
    closest_row = min(rows, key=lambda r: abs(int(r[2]) - target_bs) + abs(int(r[3]) - kv_len))
    mfu = float(closest_row[5])

    return round(mfu, 3)


def get_attn_prefill_mfu(config, seq_len, device_type, tp_size):
    gpu = gpu_map[device_type]
    tp_num_heads = config.num_attention_heads // tp_size
    if config.attn_type == "MHA/GQA":
        tp_num_kv_heads = config.num_key_value_heads // tp_size
        head_dim = config.head_dim
        file_name = f"bench_data/mha/prefill/{device_type.lower()}/{tp_num_heads}-{tp_num_kv_heads}-{head_dim}.csv"
    elif config.attn_type == "MLA":
        head_dim = f"{config.qk_nope_head_dim}-{config.qk_rope_head_dim}"
        file_name = f"bench_data/mla/prefill/{device_type.lower()}/{tp_num_heads}-{head_dim}.csv"
    if not os.path.exists(file_name):
        print(f"Warning: {file_name} not exist.")
        return 0.9

    # row: dtype,seq_len,latecy_us,mfu
    rows = list()
    with open(file_name, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            rows.append(row)

    mfu = gpu.mfu
    # mfu_seq_len = 1
    for row in rows:
        sql = int(row[1])
        if sql <= seq_len:
            # mfu_seq_len = sql
            mfu = float(row[3])
        else:
            break

    return round(mfu, 3)


def get_groupedgemm_decode_mfu(config, target_bs, device_type, num_gpus, use_fp8, tp_size=1):
    gpu = gpu_map[device_type]
    file_name = f"bench_data/grouped_gemm/decode/{device_type.lower()}/data.csv"
    if not os.path.exists(file_name):
        print(f"warning: {file_name} not exists")
        return gpu.mfu, gpu.mfu

    # row: num_experts,num_gpus,num_local_experts,topk,hidden_size,intermediate_size,batch_size_per_gpu,tokens_per_expert,up_proj_us,up_mfu,down_proj_us,down_mfu
    ep_size = num_gpus // tp_size
    expected_num_local_experts = config.num_routed_experts // ep_size
    rows = list()
    with open(file_name, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if int(row[0]) != config.num_routed_experts:
                continue
            if int(row[1]) != num_gpus:
                continue
            if int(row[2]) != expected_num_local_experts:
                continue
            if int(row[3]) != config.num_experts_per_tok:
                continue
            if int(row[4]) != config.hidden_size:
                continue
            if int(row[5]) != config.intermediate_size // tp_size:
                continue
            rows.append(row)

    if len(rows) == 0:
        print("Warning: grouped_gemm decode mfu not found, will use default mfu.")
        return gpu.mfu, gpu.mfu

    closest_row = min(rows, key=lambda r: abs(int(r[6]) - target_bs))
    mfu1 = float(closest_row[9])
    mfu2 = float(closest_row[11])

    return round(mfu1, 3), round(mfu2, 3)

def get_groupedgemm_prefill_mfu(config, seq_len, device_type, num_gpus, use_fp8, tp_size=1):
    gpu = gpu_map[device_type]
    file_name = f"bench_data/grouped_gemm/prefill/{device_type.lower()}/data.csv"
    if not os.path.exists(file_name):
        print(f"Warning: {file_name} not exists")
        return gpu.mfu, gpu.mfu

    # row: num_experts,num_gpus,num_local_experts,topk,hidden_size,intermediate_size,seq_len_per_gpu,tokens_per_expert,up_proj_us,up_mfu,down_proj_us,down_mfu
    ep_size = num_gpus // tp_size
    expected_num_local_experts = config.num_routed_experts // ep_size
    rows = list()
    with open(file_name, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if int(row[0]) != config.num_routed_experts:
                continue
            if int(row[1]) != num_gpus:
                continue
            if int(row[2]) != expected_num_local_experts:
                continue
            if int(row[3]) != config.num_experts_per_tok:
                continue
            if int(row[4]) != config.hidden_size:
                continue
            if int(row[5]) != config.intermediate_size // tp_size:
                continue
            rows.append(row)

    if len(rows) == 0:
        print("Warning: grouped_gemm prefill mfu not found, will use default mfu.")

    mfu1 = gpu.mfu
    mfu2 = gpu.mfu
    for row in rows:
        sql = int(row[6])
        if sql <= seq_len:
            mfu1 = float(row[9])
            mfu2 = float(row[11])
        else:
            break

    return round(mfu1, 3), round(mfu2, 3)


def get_gemm_mfu(device_type, m, k, n):
    gpu = gpu_map[device_type]
    file_name = f"bench_data/gemm/{device_type.lower()}/data.csv"
    if not os.path.exists(file_name):
        print(f"Warning: {file_name} not exists")
        return gpu.mfu

    mfu_k = 0
    mfu_n = 0
    dist = 1e9
    # row: m,k,n,latency_us,mfu
    rows = list()
    with open(file_name, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            k_ = int(row[1])
            n_ = int(row[2])
            if k_ < k or n_ < n:
                continue
            if (k - k_) ** 2 + (n - n_) ** 2 < dist:
                dist = (k - k_) ** 2 + (n - n_) ** 2
                mfu_k = k_
                mfu_n = n_
            rows.append(row)

    mfu = gpu.mfu
    for row in rows:
        m_ = int(row[0])
        k_ = int(row[1])
        n_ = int(row[2])
        if k_ == mfu_k and n_ == mfu_n and m_ <= m:
            mfu = float(row[4])

    return round(mfu, 3)


def get_linear_attn_prefill_latency(config, seq_len, device_type):
    file_name = f"bench_data/gdn/prefill/{device_type.lower()}/{config.linear_conv_kernel_dim}-{config.linear_num_key_heads}-{config.linear_key_head_dim}-{config.linear_num_value_heads}-{config.linear_value_head_dim}.csv"
    if not os.path.exists(file_name):
        assert False, f"Error: {file_name} not exist."

    # seq_len,conv,kkt,tril,wu,chunk_gdn,chunk_o
    rows = list()
    with open(file_name, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            rows.append(row)
    idx = 0
    diff = 1e9
    for i in range(len(rows)):
        sq = int(rows[i][0])
        if abs(sq - seq_len) < diff:
            diff = abs(sq - seq_len)
            idx = i
    ratio = int(rows[idx][0]) / seq_len
    t = 0
    for i in range(1, len(rows[idx])):
        t += float(rows[idx][i]) / ratio
    return t  # latency in us


def get_linear_attn_decode_latency(config, batchsize, device_type):
    file_name = f"bench_data/gdn/decode/{device_type.lower()}/{config.linear_conv_kernel_dim}-{config.linear_num_key_heads}-{config.linear_key_head_dim}-{config.linear_num_value_heads}-{config.linear_value_head_dim}.csv"
    if not os.path.exists(file_name):
        assert False, f"Error: {file_name} not exist."

    # batchsize,conv,gdn_update
    rows = list()
    with open(file_name, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            rows.append(row)
    idx = 0
    diff = 1e9
    for i in range(len(rows)):
        bs = int(rows[i][0])
        if abs(bs - batchsize) < diff:
            diff = abs(bs - batchsize)
            idx = i
    ratio = int(rows[idx][0]) / batchsize
    t = 0
    for i in range(1, len(rows[idx])):
        t += float(rows[idx][i]) / ratio
    return t  # latency in us
