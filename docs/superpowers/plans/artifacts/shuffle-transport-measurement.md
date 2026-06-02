# Shuffle Transport Measurement

## Hardware

- GPU model:
- GPU count per node:
- Node count:
- Interconnect:
- UCX settings:
- CUDA version:
- RAPIDS branch:

## Query

- Benchmark:
- Scale:
- Input path:
- Engine mode:
- Target partition size:
- Spill device limit:

## Commands

```bash
RAPIDSMPF_STATISTICS=True python <benchmark-command>
```

## Metrics

- wall_seconds:
- shuffle-payload-send:
- shuffle-payload-recv:
- shuffle-chunks-submit:
- shuffle-chunks-recv:
- metadata-payload-message-send:
- metadata-payload-message-recv:
- metadata-payload-payload-send:
- metadata-payload-payload-recv:
- cudf-partition-packed-bytes:
- cudf-unpack-output-bytes:

## Notes

- GPUDirect RDMA active:
- NVLink same-node GPU pairs:
- host staging observed:
