import argparse

from config.model_config import ModelConfig
from models.hybrid_model import HybridModel
from models.model import Model


def main(args):
    config = ModelConfig(args.config_path)

    print("\n{s:{c}^{n}}".format(s=" Simulator Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Device type:", args.device_type))
    print("{:<40} {:<10}".format("World size:", args.world_size))
    print("{:<40} {:<10}".format("Attn type:", config.attn_type))
    print("{:<40} {:<10}".format("Use FP8 GEMM:", args.use_fp8_gemm))
    print("{:<40} {:<10}".format("Use FP8 KV:", args.use_fp8_kv))

    if config.is_hybrid_linear:
        model = HybridModel(args, config)
    else:
        model = Model(args, config)
    model.print_weights_info()
    model.print_kvcache_info()
    model.print_flops_info()

    if not args.decode_only:
        model.prefill()

    if not args.prefill_only:
        model.decoding()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=str,
        help="The path of the hf model config.json",
        required=True,
    )
    parser.add_argument(
        "--device-type",
        type=str,
        default="H20",
        choices=["H20", "H800"],
        help="Device type",
    )
    parser.add_argument("--world-size", type=int, default=1, help="Num of GPUs")
    parser.add_argument("--num-nodes", type=int, default=1, help="Num of nodes")
    parser.add_argument(
        "--max-prefill-tokens", type=int, default=4096, help="Max prefill tokens"
    )
    parser.add_argument(
        "--decode-bs",
        type=int,
        help="Decoding batchsize. If not specified, bs = tgs * tpot.",
    )
    parser.add_argument(
        "--target-tgs", type=float, default=2560, help="Target tokens/s per GPU"
    )
    parser.add_argument("--target-tpot", type=float, default=50, help="TPOT in ms")
    parser.add_argument(
        "--target-isl", type=int, default=4096, help="Input sequence length, in tokens"
    )
    parser.add_argument(
        "--target-osl", type=int, default=2048, help="Output sequence length, in tokens"
    )
    parser.add_argument("--use-fp8-gemm", action="store_true", help="Use fp8 gemm")
    parser.add_argument("--use-fp8-kv", action="store_true", help="Use fp8 kvcache")
    parser.add_argument("--enable-deepep", action="store_true", help="Enable DeepEP")
    parser.add_argument(
        "--enable-tbo", action="store_true", help="Enable two batch overlap"
    )
    parser.add_argument(
        "--sm-ratio",
        type=float,
        default=108 / 132,
        help="In TBO DeepEP normal mode, the SM ratio used for computation",
    )
    parser.add_argument(
        "--prefill-only", action="store_true", help="Only simulate prefill"
    )
    parser.add_argument(
        "--decode-only", action="store_true", help="Only simulate decoding"
    )
    args = parser.parse_args()
    main(args)
