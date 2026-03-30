from argparse import ArgumentParser
import sys
import json

CONFIG_FILE = None
VERBOSE = True
NO_CUDA = False
SEED = None

PROBLEM = "vrp"
CUST_COUNT = 10
VEH_COUNT = 2
VEH_CAPA = 200
VEH_SPEED = 1
HORIZON = 480
MIN_CUST_COUNT = None
LOC_RANGE = (0,101)
DEM_RANGE = (5,41)
DUR_RANGE = (10,31)
TW_RATIO = (0.25,0.5,0.75,1.0)
TW_RANGE = (30,91)
DEG_OF_DYN = (0.1,0.25,0.5,0.75)
APPEAR_EARLY_RATIO = (0.0,0.5,0.75,1.0)

PEND_COST = 2
PEND_GROWTH = None
LATE_COST = 1
LATE_GROWTH = None
SPEED_VAR = 0.1
LATE_PROB = 0.05
SLOW_DOWN = 0.5
LATE_VAR = 0.2

MODEL_SIZE = 128
LAYER_COUNT = 3
HEAD_COUNT = 8
FF_SIZE = 512
TANH_XPLOR = 10
EDGE_FEAT_SIZE = 8
CUST_K = None
MEMORY_SIZE = None
LOOKAHEAD_HIDDEN = 128
MODEL_DROPOUT = 0.1
ADAPTIVE_DEPTH = False
ADAPTIVE_MIN_LAYERS = 1
ADAPTIVE_EASY_RATIO = 0.7
LATENT_BOTTLENECK = False
LATENT_TOKENS = 32
LATENT_MIN_NODES = 64
USE_EDGE_FEATURES = True
USE_MEMORY = True
USE_OWNERSHIP = True
USE_LOOKAHEAD = True
FUSION_MODE = "mlp"
LINEAR_FUSION_WEIGHTS = (1.0, 1.0, 1.0)
ABLATION_PROFILE = "none"

EPOCH_COUNT = 20
ITER_COUNT = 1000
MINIBATCH_SIZE = 512
BASE_LR = 0.0001
LR_DECAY = None
WEIGHT_DECAY = 1e-5
MAX_GRAD_NORM = 2
GRAD_NORM_DECAY = None
LOSS_USE_CUMUL = False
AMP = False
NUM_WORKERS = 4
PIN_MEMORY = True
SMOKE_TEST = False
RESOURCE_SAFE = False
MAX_VRAM_FRACTION = None

BASELINE = "none"
ROLLOUT_COUNT = 3
ROLLOUT_THRESHOLD = 0.05
CRITIC_USE_QVAL = False
CRITIC_LR = 0.001
CRITIC_DECAY = None
ADV_NORM = True
ENTROPY_COEF = 0.01

TEST_BATCH_SIZE = 128

PPO_EPOCHS = 4
PPO_CLIP_RANGE = 0.2
PPO_VALUE_COEF = 0.5
PPO_ENTROPY_COEF = 0.01
PPO_GAMMA = 1.0
PPO_GAE_LAMBDA = 1.0
PPO_ADV_NORM = True
PPO_TARGET_KL = None

OUTPUT_DIR = None
RESUME_STATE = None
MODEL_WEIGHT = None
CHECKPOINT_PERIOD = 5


def _apply_ablation_profile(args):
        profile = getattr(args, "ablation_profile", "none")
        if profile in (None, "none"):
                return args

        profiles = {
                # full COAST-style model
                "coast": {
                        "use_edge_features": True,
                        "use_memory": True,
                        "use_ownership": True,
                        "use_lookahead": True,
                        "fusion_mode": "mlp",
                        "linear_fusion_weights": (1.0, 1.0, 1.0),
                },
                # B0-None in current implementation scope
                "b0": {
                        "use_edge_features": True,
                        "use_memory": False,
                        "use_ownership": False,
                        "use_lookahead": False,
                        "fusion_mode": "mlp",
                        "linear_fusion_weights": (1.0, 1.0, 1.0),
                },
                # B1-Memory
                "b1": {
                        "use_edge_features": True,
                        "use_memory": True,
                        "use_ownership": False,
                        "use_lookahead": False,
                        "fusion_mode": "mlp",
                        "linear_fusion_weights": (1.0, 1.0, 1.0),
                },
                # B3-Look
                "b3": {
                        "use_edge_features": True,
                        "use_memory": False,
                        "use_ownership": False,
                        "use_lookahead": True,
                        "fusion_mode": "mlp",
                        "linear_fusion_weights": (1.0, 1.0, 1.0),
                },
                # B5-Linear fusion
                "b5": {
                        "use_edge_features": True,
                        "use_memory": True,
                        "use_ownership": True,
                        "use_lookahead": True,
                        "fusion_mode": "linear",
                        "linear_fusion_weights": (1.0, 1.0, 1.0),
                },
                # Edge-off ablation
                "edgeoff": {
                        "use_edge_features": False,
                        "use_memory": True,
                        "use_ownership": True,
                        "use_lookahead": True,
                        "fusion_mode": "mlp",
                        "linear_fusion_weights": (1.0, 1.0, 1.0),
                },
                # aliases for current-scope matrix
                "a0": {
                        "use_edge_features": True,
                        "use_memory": False,
                        "use_ownership": False,
                        "use_lookahead": False,
                        "fusion_mode": "mlp",
                        "linear_fusion_weights": (1.0, 1.0, 1.0),
                },
                "a1": {
                        "use_edge_features": True,
                        "use_memory": True,
                        "use_ownership": False,
                        "use_lookahead": False,
                        "fusion_mode": "mlp",
                        "linear_fusion_weights": (1.0, 1.0, 1.0),
                },
                "a3": {
                        "use_edge_features": True,
                        "use_memory": False,
                        "use_ownership": False,
                        "use_lookahead": True,
                        "fusion_mode": "mlp",
                        "linear_fusion_weights": (1.0, 1.0, 1.0),
                },
                "a4": {
                        "use_edge_features": False,
                        "use_memory": True,
                        "use_ownership": True,
                        "use_lookahead": True,
                        "fusion_mode": "mlp",
                        "linear_fusion_weights": (1.0, 1.0, 1.0),
                },
                "a9": {
                        "use_edge_features": True,
                        "use_memory": True,
                        "use_ownership": True,
                        "use_lookahead": True,
                        "fusion_mode": "linear",
                        "linear_fusion_weights": (1.0, 1.0, 1.0),
                },
        }

        selected = profiles.get(profile)
        if selected is None:
                raise ValueError("Unknown ablation profile '{}'".format(profile))

        for key, value in selected.items():
                setattr(args, key, value)

        return args


def write_config_file(args, output_file):
    with open(output_file, 'w') as f:
        json.dump(vars(args), f, indent = 4)


def parse_args(argv = None):
    parser = ArgumentParser()

    parser.add_argument("--config-file", "-f", type = str, default = CONFIG_FILE)
    parser.add_argument("--verbose", "-v", action = "store_true", default = VERBOSE)
    parser.add_argument("--no-cuda", action = "store_true", default = NO_CUDA)
    parser.add_argument("--rng-seed", type = int, default = SEED)

    group = parser.add_argument_group("Data generation parameters")
    group.add_argument("--problem-type", "-p", type = str,
            choices = ["vrp", "vrptw", "svrptw", "sdvrptw" , "dvrptw"], default = PROBLEM)
    group.add_argument("--customers-count", "-n", type = int, default = CUST_COUNT)
    group.add_argument("--vehicles-count", "-m", type = int, default = VEH_COUNT)
    group.add_argument("--veh-capa", type = int, default = VEH_CAPA)
    group.add_argument("--veh-speed", type = int, default = VEH_SPEED)
    group.add_argument("--horizon", type = int, default = HORIZON)
    group.add_argument("--min-cust-count", type = int, default = MIN_CUST_COUNT)
    group.add_argument("--loc-range", type = int, nargs = 2, default = LOC_RANGE)
    group.add_argument("--dem-range", type = int, nargs = 2, default = DEM_RANGE)
    group.add_argument("--dur-range", type = int, nargs = 2, default = DUR_RANGE)
    group.add_argument("--tw-ratio", type = float, nargs = '*', default = TW_RATIO)
    group.add_argument("--tw-range", type = int, nargs = 2, default = TW_RANGE)
    group.add_argument("--deg-of-dyna", type = float, nargs = '*', default = DEG_OF_DYN)
    group.add_argument("--appear-early-ratio", type = float, nargs = '*', default = APPEAR_EARLY_RATIO)

    group = parser.add_argument_group("VRP Environment parameters")
    group.add_argument("--pending-cost", type = float, default = PEND_COST)
    group.add_argument("--pend-cost-growth", type = float, default = PEND_GROWTH)
    group.add_argument("--late-cost", type = float, default = LATE_COST)
    group.add_argument("--late-cost-growth", type = float, default = LATE_GROWTH)
    group.add_argument("--speed-var", type = float, default = SPEED_VAR)
    group.add_argument("--late-prob", type = float, default = LATE_PROB)
    group.add_argument("--slow-down", type = float, default = SLOW_DOWN)
    group.add_argument("--late-var", type = float, default = LATE_VAR)

    group = parser.add_argument_group("Model parameters")
    group.add_argument("--model-size", "-s", type = int, default = MODEL_SIZE)
    group.add_argument("--layer-count", type = int, default = LAYER_COUNT)
    group.add_argument("--head-count", type = int, default = HEAD_COUNT)
    group.add_argument("--ff-size", type = int, default = FF_SIZE)
    group.add_argument("--tanh-xplor", type = float, default = TANH_XPLOR)
    group.add_argument("--edge-feat-size", type = int, default = EDGE_FEAT_SIZE)
    group.add_argument("--cust-k", type = int, default = CUST_K)
    group.add_argument("--memory-size", type = int, default = MEMORY_SIZE)
    group.add_argument("--lookahead-hidden", type = int, default = LOOKAHEAD_HIDDEN)
    group.add_argument("--dropout", type = float, default = MODEL_DROPOUT)
    group.add_argument("--adaptive-depth", action = "store_true", default = ADAPTIVE_DEPTH)
    group.add_argument("--adaptive-min-layers", type = int, default = ADAPTIVE_MIN_LAYERS)
    group.add_argument("--adaptive-easy-ratio", type = float, default = ADAPTIVE_EASY_RATIO)
    group.add_argument("--latent-bottleneck", action = "store_true", default = LATENT_BOTTLENECK)
    group.add_argument("--latent-tokens", type = int, default = LATENT_TOKENS)
    group.add_argument("--latent-min-nodes", type = int, default = LATENT_MIN_NODES)
    group.add_argument(
            "--ablation-profile",
            type = str,
            choices = ["none", "coast", "b0", "b1", "b3", "b5", "edgeoff", "a0", "a1", "a3", "a4", "a9"],
            default = ABLATION_PROFILE,
            help = "Apply a predefined ablation configuration with one flag",
    )
    group.add_argument("--disable-edge-features", action = "store_false", dest = "use_edge_features", default = USE_EDGE_FEATURES)
    group.add_argument("--disable-memory", action = "store_false", dest = "use_memory", default = USE_MEMORY)
    group.add_argument("--disable-ownership", action = "store_false", dest = "use_ownership", default = USE_OWNERSHIP)
    group.add_argument("--disable-lookahead", action = "store_false", dest = "use_lookahead", default = USE_LOOKAHEAD)
    group.add_argument("--fusion-mode", type = str, choices = ["mlp", "linear"], default = FUSION_MODE)
    group.add_argument("--linear-fusion-weights", type = float, nargs = 3, default = LINEAR_FUSION_WEIGHTS,
            help = "Weights for linear fusion mode: att ownership lookahead")

    group = parser.add_argument_group("Training parameters")
    group.add_argument("--epoch-count", "-e", type = int, default = EPOCH_COUNT)
    group.add_argument("--iter-count", "-i", type = int, default = ITER_COUNT)
    group.add_argument("--batch-size", "-b", type = int, default = MINIBATCH_SIZE)
    group.add_argument("--learning-rate", "-r", type = float, default = BASE_LR)
    group.add_argument("--rate-decay", "-d", type = float, default = LR_DECAY)
    group.add_argument("--weight-decay", type = float, default = WEIGHT_DECAY)
    group.add_argument("--max-grad-norm", type = float, default = MAX_GRAD_NORM)
    group.add_argument("--grad-norm-decay", type = float, default = GRAD_NORM_DECAY)
    group.add_argument("--loss-use-cumul", action = "store_true", default = LOSS_USE_CUMUL)
    group.add_argument("--amp", action = "store_true", default = AMP)
    group.add_argument("--num-workers", type = int, default = NUM_WORKERS)
    group.add_argument("--pin-memory", action = "store_true", default = PIN_MEMORY)
    group.add_argument("--smoke-test", action = "store_true", default = SMOKE_TEST,
            help = "Apply a tiny low-resource profile for fast sanity checks")
    group.add_argument("--resource-safe", action = "store_true", default = RESOURCE_SAFE,
            help = "Enable runtime safeguards against hardware overuse (OOM handling, conservative cuDNN)")
    group.add_argument("--max-vram-fraction", type = float, default = MAX_VRAM_FRACTION,
            help = "Limit this process to a fraction of GPU VRAM (0,1], e.g. 0.9")

    group = parser.add_argument_group("Baselines parameters")
    group.add_argument("--baseline-type", type = str,
            choices = ["none", "nearnb", "rollout", "critic"], default = BASELINE)
    group.add_argument("--rollout-count", type = int, default = ROLLOUT_COUNT)
    group.add_argument("--rollout-threshold", type = float, default = ROLLOUT_THRESHOLD)
    group.add_argument("--critic-use-qval", action = "store_true", default = CRITIC_USE_QVAL)
    group.add_argument("--critic-rate", type = float, default = CRITIC_LR)
    group.add_argument("--critic-decay", type = float, default = CRITIC_DECAY)
    group.add_argument("--adv-norm", action = "store_true", default = ADV_NORM,
            help = "Normalize advantages to zero mean / unit std per batch (recommended for critic baseline)")
    group.add_argument("--entropy-coef", type = float, default = ENTROPY_COEF,
            help = "Entropy regularization coefficient (e.g. 0.01) to prevent premature policy collapse")

    group = parser.add_argument_group("Testing parameters")
    group.add_argument("--test-batch-size", type = int, default = TEST_BATCH_SIZE)

    group = parser.add_argument_group("PPO parameters")
    group.add_argument("--ppo-epochs", type = int, default = PPO_EPOCHS)
    group.add_argument("--ppo-clip-range", type = float, default = PPO_CLIP_RANGE)
    group.add_argument("--ppo-value-coef", type = float, default = PPO_VALUE_COEF)
    group.add_argument("--ppo-entropy-coef", type = float, default = PPO_ENTROPY_COEF)
    group.add_argument("--ppo-gamma", type = float, default = PPO_GAMMA)
    group.add_argument("--ppo-gae-lambda", type = float, default = PPO_GAE_LAMBDA)
    group.add_argument("--ppo-adv-norm", action = "store_true", default = PPO_ADV_NORM)
    group.add_argument("--no-ppo-adv-norm", action = "store_false", dest = "ppo_adv_norm")
    group.add_argument("--ppo-target-kl", type = float, default = PPO_TARGET_KL)

    group = parser.add_argument_group("Checkpointing")
    group.add_argument("--output-dir", "-o", type = str, default = OUTPUT_DIR)
    group.add_argument("--checkpoint-period", "-c", type = int, default = CHECKPOINT_PERIOD)
    group.add_argument("--resume-state", type = str, default = RESUME_STATE)
    group.add_argument("--model-weight", type = str, default = MODEL_WEIGHT)

    args = parser.parse_args(argv)
    if args.config_file is not None:
        with open(args.config_file) as f:
            parser.set_defaults(**json.load(f))

    args = parser.parse_args(argv)
    return _apply_ablation_profile(args)
