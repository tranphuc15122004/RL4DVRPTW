# DVRPTW Inference Analysis & Implementation

## Phân tích kiến trúc

### 1. AM Inference (`am/infer.py`)

- Input: `.pyth` (torch-serialized), `.csv` (qua `DVRPTW_Dataset.from_csv`), hoặc generate on-the-fly
- Model: `AM_DVRPTW` (Attention Model) — `rl4co` AttentionModelPolicy
- Output: routes (list of lists), costs (tensor), optional JSON save
- Pipeline: load data → normalize → build env → init model → warmup → load weights → infer → verify → report

### 2. PolyNet Inference (`polynet/infer.py`)

- Cấu trúc gần như giống hệt AM, chỉ khác model class (`PolyNet_DVRPTW`) và tham số `cust_k`
- Code trùng lặp ~90% với `am/infer.py`

### 3. Dataset hierarchy

| Class | CUST_FEAT_SIZE | Features |
|-------|-------|----------|
| `VRP_Dataset` | 3 | x, y, demand |
| `VRPTW_Dataset` | 6 | + open, close, servicetime |
| `DVRPTW_Dataset` | 7 | + appearance_time |

### 4. Environment hierarchy

| Class | Key additions |
|-------|---------------|
| `VRP_Environment` | base: capacity, pending cost |
| `VRPTW_Environment` | + late_cost, time windows |
| `DVRPTW_Environment` | + dynamic customer reveal |

### 5. Dữ liệu

- `.pyth` files: `dvrptw_n{100,200,400,50}m{5,10,20,3}_10240.pyth`
- CSV files: `data/datasets/{100,200,400,1000}/` — benchmark instances
  - Cột: x, y, demand, open, close, servicetime, drone_serve, time
  - `nodes_from_csv` đọc 7 cột (bỏ drone_serve), CUST_FEAT_SIZE = 7

## Đã triển khai

### Module dùng chung — `utils/infer_utils.py`

- `build_dataset()` — load từ `.pyth`, `.csv`, hoặc generate
- `init_am_model()` / `init_polynet_model()` — khởi tạo model
- `load_model_weights()` — load checkpoint
- `run_inference()` / `run_single_inference()` — chạy inference
- `full_inference_pipeline()` — pipeline hoàn chỉnh (dùng cho cả AM và PolyNet)
- `verify_routes_cost()`, `check_route_constraints()`, `compute_cost_components()` — diagnostics
- `discover_csv_files()` — tìm CSV files trong thư mục
- `save_json()`, `add_infer_args()`, `parse_infer_args()`

### Single-instance inference

- **AM**: `am/infer_single.py` — infer 1 CSV/Pyth instance với output chi tiết (từng xe, diagnostics)
- **PolyNet**: `polynet/infer_single.py` — tương tự cho PolyNet

### Batch inference — `infer_batch.py`

- Hỗ trợ: AM, PolyNet, hoặc **compare** (cả 2 trên cùng data)
- Input: `--data-file` (1 .pyth), `--data-csv` (1 CSV), `--csv-dir` (batch CSV), `--pyth-dir` (batch .pyth)
- Output: per-file JSON + aggregated CSV
- Filtering: `--max-files`, `--file-pattern`

### Shell scripts

- `run_am_infer.sh` — AM inference (MODE: single-csv, single-pyth, batch-csv, batch-pyth)
- `run_polynet_infer.sh` — PolyNet inference
- `run_compare_infer.sh` — So sánh AM vs PolyNet

## Usage examples

```bash
# AM — single CSV
python3 am/infer_single.py \
  --data-csv data/datasets/100/h100c101.csv \
  --model-weight data/_AM/chkpt_best.pyth \
  --customers-count 100 --vehicles-count 5 --veh-capa 1300 --veh-speed 1

# PolyNet — single CSV
python3 polynet/infer_single.py \
  --data-csv data/datasets/100/h100c101.csv \
  --model-weight data/_PolyNet/chkpt_best.pyth \
  --customers-count 100 --vehicles-count 5 --veh-capa 1300 --veh-speed 1

# AM — auto-configure from training args.json (không cần --customers-count, --vehicles-count,...)
python3 am/infer_single.py \
  --data-csv data/datasets/100/h100c101.csv \
  --model-weight data/_AM/chkpt_best.pyth \
  --model-args data/_AM/args.json

# AM — batch CSV với auto-config
python3 infer_batch.py --model am \
  --model-weight data/_AM/chkpt_best.pyth \
  --csv-dir data/datasets/100 \
  --model-args data/_AM/args.json \
  --output-dir output/batch_am_100

# Compare AM vs PolyNet (mỗi model dùng args.json riêng)
python3 infer_batch.py --model compare \
  --am-weight data/_AM/chkpt_best.pyth \
  --polynet-weight data/_PolyNet/chkpt_best.pyth \
  --csv-dir data/datasets/100 \
  --am-args data/_AM/args.json \
  --polynet-args data/_PolyNet/args.json \
  --output-dir output/compare_100

# Using shell scripts with auto-config
MODE=single-csv DATA_PATH=data/datasets/100/h100c101.csv MODEL_ARGS=data/_AM/args.json ./run_am_infer.sh
MODEL_ARGS=data/_PolyNet/args.json MODE=batch-csv DATA_PATH=data/datasets/100 ./run_polynet_infer.sh
AM_ARGS=data/_AM/args.json POLYNET_ARGS=data/_PolyNet/args.json ./run_compare_infer.sh
```

## `--model-args` feature

Các script inference (cả standalone `am/infer.py`, `polynet/infer.py` và batch `infer_batch.py`) đều hỗ trợ `--model-args` / `--am-args` / `--polynet-args` để tự động nạp tham số từ training `args.json`.

**Cách hoạt động:**

1. Đọc file JSON chứa toàn bộ config huấn luyện
2. Trích xuất các tham số liên quan đến inference (model architecture, problem config, environment)
3. So sánh với module-level defaults trong `utils/_args.py`; nếu giá trị hiện tại trùng với default → thay bằng giá trị từ JSON
4. Các flag CLI tường minh luôn được ưu tiên hơn JSON

**Tham số được nạp từ JSON:**

- Model: `model_size`, `layer_count`, `head_count`, `ff_size`, `tanh_xplor`, `cust_k`, `dropout`, `ablation_profile`, fusion params
- Problem: `customers_count`, `vehicles_count`, `veh_capa`, `veh_speed`, `horizon`, `loc_range`, `dem_range`, `tw_ratio`, `tw_range`, `deg_of_dyna`, `appear_early_ratio`
- Environment: `pending_cost`, `late_cost`, `speed_var`, `late_prob`, `slow_down`, `late_var`

## Công việc cần làm tiếp

- [ ] Chạy thử infer cho PolyNet (cần model weight)
- [ ] Debug PolyNet inference nếu lỗi
- [ ] Kiểm tra batch CSV với nhiều files
- [ ] Thêm unit tests cho inference pipeline
