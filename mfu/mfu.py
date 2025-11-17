import csv
import os

from hardware.gpu import gpu_map


def get_attn_decode_mfu(config, target_bs, kv_len, device_type, use_fp8_kv):
    gpu = gpu_map[device_type]
    if config.attn_type == "MHA/GQA":
        head_dim = config.head_dim
        file_name = f"bench_data/mha/decode/{device_type.lower()}/{config.num_attention_heads}-{config.num_key_value_heads}-{head_dim}.csv"
    elif config.attn_type == "MLA":
        head_dim = f"{config.kv_lora_rank}-{config.qk_rope_head_dim}"
        file_name = f"bench_data/mla/decode/{device_type.lower()}/{config.num_attention_heads}-{head_dim}.csv"
    if not os.path.exists(file_name):
        print(f"warning: {file_name} not exists")
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

    mfu_bs = 1
    for row in rows:
        bs = int(row[2])
        if bs <= target_bs:
            mfu_bs = bs
        else:
            break

    mfu_kv_len = 1
    for row in rows:
        kv_l = int(row[3])
        if kv_l <= kv_len:
            mfu_kv_len = kv_l
        else:
            break

    mfu = gpu.mfu
    for row in rows:
        bs = int(row[2])
        kv_l = int(row[3])
        if bs == mfu_bs and kv_l == mfu_kv_len:
            mfu = float(row[5])

    return round(mfu, 3)


def get_attn_prefill_mfu(config, seq_len, device_type):
    gpu = gpu_map[device_type]
    if config.attn_type == "MHA/GQA":
        head_dim = config.head_dim
        file_name = f"bench_data/mha/prefill/{device_type.lower()}/{config.num_attention_heads}-{config.num_key_value_heads}-{head_dim}.csv"
    elif config.attn_type == "MLA":
        head_dim = f"{config.qk_nope_head_dim}-{config.qk_rope_head_dim}"
        file_name = f"bench_data/mla/prefill/{device_type.lower()}/{config.num_attention_heads}-{head_dim}.csv"
    if not os.path.exists(file_name):
        print(f"warning: {file_name} not exist.")
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


def get_groupedgemm_decode_mfu(config, target_bs, device_type, num_gpus, use_fp8):
    gpu = gpu_map[device_type]
    file_name = f"bench_data/grouped_gemm/decode/{device_type.lower()}/data.csv"
    if not os.path.exists(file_name):
        print(f"warning: {file_name} not exists")
        return gpu.mfu, gpu.mfu

    # row: num_experts,num_gpus,num_local_experts,topk,hidden_size,intermediate_size,batch_size_per_gpu,tokens_per_expert,up_proj_us,up_mfu,down_proj_us,down_mfu
    rows = list()
    with open(file_name, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if int(row[0]) != config.num_routed_experts:
                continue
            if int(row[1]) != num_gpus:
                continue
            if int(row[3]) != config.num_experts_per_tok:
                continue
            if int(row[4]) != config.hidden_size:
                continue
            if int(row[5]) != config.intermediate_size:
                continue
            rows.append(row)

    mfu1 = gpu.mfu
    mfu2 = gpu.mfu
    for row in rows:
        bs = int(row[6])
        if bs <= target_bs:
            mfu1 = float(row[9])
            mfu2 = float(row[11])
        else:
            break

    return round(mfu1, 3), round(mfu2, 3)


def get_groupedgemm_prefill_mfu(config, seq_len, device_type, num_gpus, use_fp8):
    gpu = gpu_map[device_type]
    file_name = f"bench_data/grouped_gemm/prefill/{device_type.lower()}/data.csv"
    if not os.path.exists(file_name):
        print(f"warning: {file_name} not exists")
        return gpu.mfu, gpu.mfu

    # row: num_experts,num_gpus,num_local_experts,topk,hidden_size,intermediate_size,seq_len_per_gpu,tokens_per_expert,up_proj_us,up_mfu,down_proj_us,down_mfu
    rows = list()
    with open(file_name, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if int(row[0]) != config.num_routed_experts:
                continue
            if int(row[1]) != num_gpus:
                continue
            if int(row[3]) != config.num_experts_per_tok:
                continue
            if int(row[4]) != config.hidden_size:
                continue
            if int(row[5]) != config.intermediate_size:
                continue
            rows.append(row)

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
        print(f"warning: {file_name} not exists")
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
        t += float(rows[idx][i]) * ratio
    return t  # latency in us
