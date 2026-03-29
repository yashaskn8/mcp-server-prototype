# services/mock_data.py
# Isolated mock data store.
#
# Separating mock data from service logic is intentional:
#   • The service layer (k8s_service.py) never references this module directly
#     when MOCK_MODE=False; it calls the real Training Operator SDK instead.
#   • Keeping all fake records here makes it trivial to add/remove test
#     scenarios without touching business logic.
#
# Data is modelled to be realistic:
#   • Pod names follow the Training Operator convention:
#       <job-name>-<framework>job-<role>-<index>
#   • Events mirror real Kubernetes event Reasons seen in production clusters
#   • Log lines reflect actual PyTorch / TF distributed training output
#   • Metrics values are plausible for the declared model architectures

from models.schema import (
    Framework, JobStatus, PodPhase, EventType,
    ResourceSpec, PodInfo, TrainingMetrics, JobEvent,
    TrainJobDetail, NamespaceSummary,
)

# ──────────────────────────────────────────────────────────────────
# Training Jobs
# ──────────────────────────────────────────────────────────────────

JOBS: dict[str, dict] = {

    # ── RUNNING — multi-node PyTorch DDP image classification ──────
    "job-001": {
        "job_id":       "job-001",
        "name":         "resnet50-imagenet",
        "namespace":    "kubeflow",
        "status":       JobStatus.RUNNING,
        "framework":    Framework.PYTORCH,
        "created_at":   "2025-03-20T08:00:00Z",
        "message":      "Training in progress — epoch 12 / 50",
        "num_workers":  4,
        "num_masters":  1,
        "resources":    ResourceSpec(cpu="8", memory="32Gi", gpu=2),
        "image":        "pytorch/pytorch:2.2-cuda12.1-cudnn8-runtime",
        "duration_mins": None,
        "completed_at": None,
        "pods": [
            PodInfo(name="resnet50-imagenet-pytorchjob-master-0",
                    role="master", phase=PodPhase.RUNNING,
                    node="gpu-node-01", restart_count=0,
                    start_time="2025-03-20T08:01:00Z"),
            PodInfo(name="resnet50-imagenet-pytorchjob-worker-0",
                    role="worker", phase=PodPhase.RUNNING,
                    node="gpu-node-02", restart_count=0,
                    start_time="2025-03-20T08:01:05Z"),
            PodInfo(name="resnet50-imagenet-pytorchjob-worker-1",
                    role="worker", phase=PodPhase.RUNNING,
                    node="gpu-node-03", restart_count=0,
                    start_time="2025-03-20T08:01:05Z"),
            PodInfo(name="resnet50-imagenet-pytorchjob-worker-2",
                    role="worker", phase=PodPhase.RUNNING,
                    node="gpu-node-04", restart_count=1,   # had a transient restart
                    start_time="2025-03-20T08:03:12Z"),
            PodInfo(name="resnet50-imagenet-pytorchjob-worker-3",
                    role="worker", phase=PodPhase.RUNNING,
                    node="gpu-node-05", restart_count=0,
                    start_time="2025-03-20T08:01:08Z"),
        ],
        "metrics": TrainingMetrics(
            step=18_432, epoch=12, total_epochs=50,
            loss=1.4873, accuracy=0.6582, learning_rate=0.001,
            gpu_utilization_pct=94.2,
            throughput_samples_per_sec=3_420.0,
            recorded_at="2025-03-20T09:14:05Z",
        ),
    },

    # ── SUCCEEDED — BERT fine-tuning on SST-2 ──────────────────────
    "job-002": {
        "job_id":       "job-002",
        "name":         "bert-sst2-finetune",
        "namespace":    "kubeflow",
        "status":       JobStatus.SUCCEEDED,
        "framework":    Framework.TENSORFLOW,
        "created_at":   "2025-03-18T14:30:00Z",
        "message":      "Job completed successfully. Best val_accuracy=0.935",
        "num_workers":  2,
        "num_masters":  1,
        "resources":    ResourceSpec(cpu="4", memory="16Gi", gpu=1),
        "image":        "tensorflow/tensorflow:2.15-gpu",
        "duration_mins": 142,
        "completed_at": "2025-03-18T16:52:00Z",
        "pods": [
            PodInfo(name="bert-sst2-finetune-tfjob-chief-0",
                    role="master", phase=PodPhase.SUCCEEDED,
                    node="gpu-node-01", restart_count=0,
                    start_time="2025-03-18T14:31:00Z"),
            PodInfo(name="bert-sst2-finetune-tfjob-worker-0",
                    role="worker", phase=PodPhase.SUCCEEDED,
                    node="gpu-node-02", restart_count=0,
                    start_time="2025-03-18T14:31:10Z"),
            PodInfo(name="bert-sst2-finetune-tfjob-worker-1",
                    role="worker", phase=PodPhase.SUCCEEDED,
                    node="gpu-node-03", restart_count=0,
                    start_time="2025-03-18T14:31:10Z"),
        ],
        "metrics": TrainingMetrics(
            step=6_274, epoch=3, total_epochs=3,
            loss=0.1843, accuracy=0.9350, learning_rate=2e-5,
            gpu_utilization_pct=87.5,
            throughput_samples_per_sec=420.0,
            recorded_at="2025-03-18T16:52:00Z",
        ),
    },

    # ── FAILED — OOM on worker during GPT-2 pretraining ───────────
    "job-003": {
        "job_id":       "job-003",
        "name":         "gpt2-124m-pretrain",
        "namespace":    "research",
        "status":       JobStatus.FAILED,
        "framework":    Framework.PYTORCH,
        "created_at":   "2025-03-19T10:00:00Z",
        "message":      "Worker-3 OOMKilled (exit 137). Exceeded 32Gi memory limit.",
        "num_workers":  8,
        "num_masters":  1,
        "resources":    ResourceSpec(cpu="16", memory="32Gi", gpu=4),
        "image":        "pytorch/pytorch:2.2-cuda12.1-cudnn8-runtime",
        "duration_mins": 31,
        "completed_at": "2025-03-19T10:31:44Z",
        "pods": [
            PodInfo(name="gpt2-124m-pretrain-pytorchjob-master-0",
                    role="master", phase=PodPhase.FAILED,
                    node="gpu-node-01", restart_count=0,
                    start_time="2025-03-19T10:01:00Z"),
            *[
                PodInfo(
                    name=f"gpt2-124m-pretrain-pytorchjob-worker-{i}",
                    role="worker",
                    phase=PodPhase.FAILED if i == 3 else PodPhase.SUCCEEDED,
                    node=f"gpu-node-{i+2:02d}",
                    restart_count=3 if i == 3 else 0,
                    start_time="2025-03-19T10:01:10Z",
                )
                for i in range(8)
            ],
        ],
        "metrics": TrainingMetrics(
            step=4_100, epoch=1, total_epochs=10,
            loss=6.824, accuracy=None, learning_rate=6e-4,
            gpu_utilization_pct=None,
            throughput_samples_per_sec=None,
            recorded_at="2025-03-19T10:28:01Z",
        ),
    },

    # ── PENDING — waiting for GPU quota ────────────────────────────
    "job-004": {
        "job_id":       "job-004",
        "name":         "stable-diffusion-lora",
        "namespace":    "ml-team",
        "status":       JobStatus.PENDING,
        "framework":    Framework.PYTORCH,
        "created_at":   "2025-03-21T09:15:00Z",
        "message":      "Waiting for GPU quota. ResourceQuota: 0 / 2 A100 GPUs available.",
        "num_workers":  2,
        "num_masters":  1,
        "resources":    ResourceSpec(cpu="8", memory="32Gi", gpu=1),
        "image":        "pytorch/pytorch:2.2-cuda12.1-cudnn8-runtime",
        "duration_mins": None,
        "completed_at": None,
        "pods": [
            PodInfo(name="stable-diffusion-lora-pytorchjob-master-0",
                    role="master", phase=PodPhase.PENDING,
                    node=None, restart_count=0, start_time=None),
        ],
        "metrics": None,
    },

    # ── SUSPENDED — manually paused by user ────────────────────────
    "job-005": {
        "job_id":       "job-005",
        "name":         "llama3-8b-sft",
        "namespace":    "research",
        "status":       JobStatus.SUSPENDED,
        "framework":    Framework.PYTORCH,
        "created_at":   "2025-03-17T06:00:00Z",
        "message":      "Job suspended by user. Resume when A100 spot capacity available.",
        "num_workers":  16,
        "num_masters":  1,
        "resources":    ResourceSpec(cpu="32", memory="128Gi", gpu=8),
        "image":        "pytorch/pytorch:2.2-cuda12.1-cudnn8-runtime",
        "duration_mins": None,
        "completed_at": None,
        "pods": [],   # Pods are deleted when a job is suspended
        "metrics": TrainingMetrics(   # Preserved from before suspension
            step=52_000, epoch=1, total_epochs=3,
            loss=2.341, accuracy=None, learning_rate=1e-4,
            gpu_utilization_pct=None,
            throughput_samples_per_sec=None,
            recorded_at="2025-03-17T12:34:00Z",
        ),
    },

    # ── RUNNING — JAX-based multi-host TPU job ─────────────────────
    "job-006": {
        "job_id":       "job-006",
        "name":         "vit-large-jax",
        "namespace":    "kubeflow",
        "status":       JobStatus.RUNNING,
        "framework":    Framework.JAX,
        "created_at":   "2025-03-21T07:00:00Z",
        "message":      "Training in progress — step 8,200 / 100,000",
        "num_workers":  4,
        "num_masters":  1,
        "resources":    ResourceSpec(cpu="16", memory="64Gi", gpu=4),
        "image":        "gcr.io/tpu-pytorch/xla:nightly",
        "duration_mins": None,
        "completed_at": None,
        "pods": [
            PodInfo(name=f"vit-large-jax-mpijob-{'master' if i == 0 else 'worker'}-{i}",
                    role="master" if i == 0 else "worker",
                    phase=PodPhase.RUNNING,
                    node=f"tpu-node-{i+1:02d}",
                    restart_count=0,
                    start_time="2025-03-21T07:01:00Z")
            for i in range(5)
        ],
        "metrics": TrainingMetrics(
            step=8_200, epoch=None, total_epochs=None,
            loss=3.112, accuracy=0.4210, learning_rate=0.0003,
            gpu_utilization_pct=99.1,
            throughput_samples_per_sec=1_100.0,
            recorded_at="2025-03-21T09:10:00Z",
        ),
    },
}


# ──────────────────────────────────────────────────────────────────
# Pod Logs  (realistic distributed-training output)
# ──────────────────────────────────────────────────────────────────

LOGS: dict[str, list[str]] = {
    "job-001": [
        "[2025-03-20 08:01:00] INFO  master-0 | Initialising PyTorch DDP — world_size=5",
        "[2025-03-20 08:01:02] INFO  master-0 | NCCL backend initialised (version 2.18.3)",
        "[2025-03-20 08:01:05] INFO  master-0 | ImageNet-1k: 1,281,167 train / 50,000 val samples",
        "[2025-03-20 08:01:06] INFO  master-0 | Model: ResNet-50 (25.6 M parameters)",
        "[2025-03-20 08:06:14] INFO  worker-0 | Epoch  1/50 | step    384 | loss=4.2310 | acc=0.021 | lr=1.00e-02 | 698 samples/s",
        "[2025-03-20 08:12:31] INFO  worker-0 | Epoch  2/50 | step    768 | loss=3.8742 | acc=0.087 | lr=1.00e-02 | 702 samples/s",
        "[2025-03-20 08:19:44] INFO  worker-0 | Epoch  3/50 | step  1,152 | loss=3.3019 | acc=0.201 | lr=1.00e-02 | 698 samples/s",
        "[2025-03-20 08:43:55] INFO  worker-2 | Container restarted (exit 137 — transient SIGKILL)",
        "[2025-03-20 08:44:10] INFO  worker-2 | Restored from step 1,152 checkpoint. Resuming.",
        "[2025-03-20 09:10:42] INFO  worker-0 | Epoch 11/50 | step 16,896 | loss=1.5618 | acc=0.641 | lr=1.00e-03 | 3,412 samples/s",
        "[2025-03-20 09:14:05] INFO  worker-0 | Epoch 12/50 | step 18,432 | loss=1.4873 | acc=0.658 | lr=1.00e-03 | 3,421 samples/s",
    ],
    "job-002": [
        "[2025-03-18 14:31:00] INFO  chief-0 | TF distributed strategy: MirroredStrategy (2 GPUs)",
        "[2025-03-18 14:32:10] INFO  chief-0 | Loading bert-base-uncased from HuggingFace hub",
        "[2025-03-18 14:33:45] INFO  chief-0 | SST-2: 67,349 train / 872 validation examples",
        "[2025-03-18 14:34:01] INFO  chief-0 | Epoch 1/3 start",
        "[2025-03-18 15:12:00] INFO  chief-0 | Epoch 1/3 | loss=0.4221 | val_acc=0.872 | 418 samples/s",
        "[2025-03-18 15:50:00] INFO  chief-0 | Epoch 2/3 | loss=0.2541 | val_acc=0.921 | 421 samples/s",
        "[2025-03-18 16:28:00] INFO  chief-0 | Epoch 3/3 | loss=0.1843 | val_acc=0.935 | 420 samples/s",
        "[2025-03-18 16:50:00] INFO  chief-0 | Saving model checkpoint to /mnt/output/bert-sst2-final.ckpt",
        "[2025-03-18 16:52:00] INFO  chief-0 | Training complete. Best val_accuracy=0.935",
    ],
    "job-003": [
        "[2025-03-19 10:01:00] INFO  master-0 | Launching PyTorch DDP — world_size=9 (1 master + 8 workers)",
        "[2025-03-19 10:01:10] INFO  master-0 | GPT-2 124M: 124,439,808 parameters",
        "[2025-03-19 10:01:12] INFO  master-0 | Dataset: OpenWebText (8.01 GB, ~9B tokens)",
        "[2025-03-19 10:05:22] INFO  worker-0 | step   100 | loss=10.312 | lr=6.00e-04 | 1,204 samples/s",
        "[2025-03-19 10:10:14] INFO  worker-0 | step   500 | loss= 7.821 | lr=6.00e-04 | 1,198 samples/s",
        "[2025-03-19 10:21:00] INFO  worker-0 | step 2,000 | loss= 6.824 | lr=6.00e-04 | 1,202 samples/s",
        "[2025-03-19 10:27:41] WARNING worker-3 | CUDA memory: allocated=29.8 GiB / 32.0 GiB (93%)",
        "[2025-03-19 10:28:01] ERROR  worker-3 | RuntimeError: CUDA out of memory. Tried to allocate 2.50 GiB.",
        "[2025-03-19 10:28:01] ERROR  worker-3 | torch.cuda.OutOfMemoryError — consider reducing batch_size or enabling gradient checkpointing.",
        "[2025-03-19 10:28:02] ERROR  master-0 | Rendezvous error: worker-3 exited with non-zero status (1).",
        "[2025-03-19 10:28:02] ERROR  master-0 | Terminating all workers. Job status → FAILED.",
    ],
    "job-004": [
        "[2025-03-21 09:15:00] INFO  scheduler | TrainJob 'stable-diffusion-lora' submitted to namespace 'ml-team'",
        "[2025-03-21 09:15:01] WARNING scheduler | ResourceQuota check failed: requested 2 GPUs, available 0.",
        "[2025-03-21 09:15:01] INFO  scheduler | Job queued. Will retry scheduling every 60 s.",
    ],
    "job-005": [
        "[2025-03-17 06:01:00] INFO  master-0 | Launching LLaMA-3 8B SFT — world_size=17",
        "[2025-03-17 06:01:30] INFO  master-0 | Flash Attention 2 enabled (CUDA 12.1)",
        "[2025-03-17 07:00:10] INFO  worker-0 | step  5,000 | loss=2.841 | lr=1.00e-04 | 680 samples/s",
        "[2025-03-17 10:00:00] INFO  worker-0 | step 30,000 | loss=2.512 | lr=1.00e-04 | 683 samples/s",
        "[2025-03-17 12:34:00] INFO  worker-0 | step 52,000 | loss=2.341 | lr=1.00e-04 | 681 samples/s",
        "[2025-03-17 12:35:00] INFO  master-0 | Suspend signal received. Saving checkpoint to /mnt/ckpt/step-52000/",
        "[2025-03-17 12:36:00] INFO  master-0 | Checkpoint saved. All workers terminated cleanly.",
    ],
    "job-006": [
        "[2025-03-21 07:01:00] INFO  master-0 | JAX distributed init — process_count=5, local_device_count=4",
        "[2025-03-21 07:01:05] INFO  master-0 | ViT-Large/16: 307,337,728 parameters",
        "[2025-03-21 07:01:08] INFO  master-0 | JIT-compiling model … (first step may be slow)",
        "[2025-03-21 07:03:22] INFO  master-0 | step     1 | loss=6.912 | acc=0.001 | compile_time=134s",
        "[2025-03-21 07:03:40] INFO  master-0 | step    10 | loss=6.731 | acc=0.004 | 1,098 samples/s",
        "[2025-03-21 08:20:00] INFO  master-0 | step 5,000 | loss=4.112 | acc=0.231 | 1,101 samples/s",
        "[2025-03-21 09:10:00] INFO  master-0 | step 8,200 | loss=3.112 | acc=0.421 | 1,100 samples/s",
    ],
}


# ──────────────────────────────────────────────────────────────────
# Kubernetes Events per job
# ──────────────────────────────────────────────────────────────────

EVENTS: dict[str, list[dict]] = {
    "job-001": [
        {"reason": "SuccessfulCreatePod", "message": "Created pod: resnet50-imagenet-pytorchjob-master-0",
         "event_type": EventType.NORMAL, "count": 1,
         "first_time": "2025-03-20T08:01:00Z", "last_time": "2025-03-20T08:01:00Z"},
        {"reason": "Pulled", "message": "Successfully pulled image pytorch/pytorch:2.2-cuda12.1-cudnn8-runtime in 42s",
         "event_type": EventType.NORMAL, "count": 5,
         "first_time": "2025-03-20T08:01:00Z", "last_time": "2025-03-20T08:01:42Z"},
        {"reason": "BackOff", "message": "Back-off restarting failed container in pod resnet50-imagenet-pytorchjob-worker-2",
         "event_type": EventType.WARNING, "count": 1,
         "first_time": "2025-03-20T08:43:55Z", "last_time": "2025-03-20T08:43:55Z"},
        {"reason": "Started", "message": "Started container pytorch-worker (restart #1)",
         "event_type": EventType.NORMAL, "count": 1,
         "first_time": "2025-03-20T08:44:10Z", "last_time": "2025-03-20T08:44:10Z"},
    ],
    "job-002": [
        {"reason": "SuccessfulCreatePod", "message": "Created pod: bert-sst2-finetune-tfjob-chief-0",
         "event_type": EventType.NORMAL, "count": 1,
         "first_time": "2025-03-18T14:31:00Z", "last_time": "2025-03-18T14:31:00Z"},
        {"reason": "TrainJobSucceeded", "message": "TrainJob bert-sst2-finetune succeeded",
         "event_type": EventType.NORMAL, "count": 1,
         "first_time": "2025-03-18T16:52:00Z", "last_time": "2025-03-18T16:52:00Z"},
    ],
    "job-003": [
        {"reason": "SuccessfulCreatePod", "message": "Created pod: gpt2-124m-pretrain-pytorchjob-master-0",
         "event_type": EventType.NORMAL, "count": 1,
         "first_time": "2025-03-19T10:01:00Z", "last_time": "2025-03-19T10:01:00Z"},
        {"reason": "OOMKilled", "message": "Container pytorch-worker in pod gpt2-124m-pretrain-pytorchjob-worker-3 OOMKilled",
         "event_type": EventType.WARNING, "count": 3,
         "first_time": "2025-03-19T10:20:00Z", "last_time": "2025-03-19T10:28:01Z"},
        {"reason": "BackoffLimitExceeded", "message": "Job has reached the specified backoff limit",
         "event_type": EventType.WARNING, "count": 1,
         "first_time": "2025-03-19T10:28:02Z", "last_time": "2025-03-19T10:28:02Z"},
        {"reason": "TrainJobFailed", "message": "TrainJob gpt2-124m-pretrain failed: worker-3 exited with code 1",
         "event_type": EventType.WARNING, "count": 1,
         "first_time": "2025-03-19T10:28:02Z", "last_time": "2025-03-19T10:28:02Z"},
    ],
    "job-004": [
        {"reason": "FailedScheduling", "message": "0/6 nodes available: insufficient nvidia.com/gpu",
         "event_type": EventType.WARNING, "count": 12,
         "first_time": "2025-03-21T09:15:01Z", "last_time": "2025-03-21T09:27:01Z"},
    ],
    "job-005": [
        {"reason": "TrainJobSuspended", "message": "TrainJob llama3-8b-sft suspended; 0 pods active",
         "event_type": EventType.NORMAL, "count": 1,
         "first_time": "2025-03-17T12:35:00Z", "last_time": "2025-03-17T12:35:00Z"},
    ],
    "job-006": [
        {"reason": "SuccessfulCreatePod", "message": "Created pod: vit-large-jax-mpijob-master-0",
         "event_type": EventType.NORMAL, "count": 1,
         "first_time": "2025-03-21T07:01:00Z", "last_time": "2025-03-21T07:01:00Z"},
    ],
}
