import json
import os
import time
import threading
import struct
from dataclasses import dataclass, asdict

import torch


@dataclass
class PacketRecord:
    timestamp: float
    round: int
    epoch: int
    batch: int

    direction: str
    client_id: int

    model: str
    stage: str

    shape: list
    dtype: str
    numel: int

    payload_bytes: int
    serialized_bytes: int

    mean: float
    std: float
    min: float
    max: float


class CommunicationMonitor:
    """
    Records logical client/core communication.

    The current Hu-Comm implementation passes tensors directly inside
    one Python process. This monitor serializes a detached CPU copy to
    measure the wire-equivalent payload size.
    """

    def __init__(
        self,
        output_dir="./Results/Communication",
        enabled=True,
        save_payloads=False,
    ):
        self.enabled = enabled
        self.save_payloads = save_payloads

        self.output_dir = output_dir
        self.jsonl_path = os.path.join(
            output_dir,
            "packets.jsonl"
        )

        self.payload_dir = os.path.join(
            output_dir,
            "payloads"
        )

        os.makedirs(output_dir, exist_ok=True)

        if save_payloads:
            os.makedirs(self.payload_dir, exist_ok=True)

        self._lock = threading.Lock()
        self.packet_id = 0

    def _next_packet_id(self):
        with self._lock:
            packet_id = self.packet_id
            self.packet_id += 1

        return packet_id

    @staticmethod
    def _serialize_tensor(tensor):
        """
        Wire-equivalent serialization.

        We store:
            magic      4 bytes
            ndim       1 byte
            shape      8 bytes per dimension
            dtype_id   1 byte
            payload    raw contiguous tensor bytes
        """

        tensor = tensor.detach().cpu().contiguous()

        dtype_map = {
            torch.float32: 1,
            torch.float64: 2,
            torch.float16: 3,
            torch.bfloat16: 4,
            torch.int64: 5,
            torch.int32: 6,
            torch.int16: 7,
            torch.int8: 8,
            torch.uint8: 9,
            torch.bool: 10,
        }

        dtype_id = dtype_map.get(tensor.dtype)

        if dtype_id is None:
            raise ValueError(
                f"Unsupported tensor dtype: {tensor.dtype}"
            )

        ndim = tensor.dim()

        header = bytearray()

        # Magic
        header.extend(b"HUC1")

        # Number of dimensions
        header.extend(struct.pack("<B", ndim))

        # Shape
        for dimension in tensor.shape:
            header.extend(struct.pack("<Q", int(dimension)))

        # dtype
        header.extend(struct.pack("<B", dtype_id))

        payload = tensor.numpy().tobytes()

        return bytes(header) + payload

    def record(
        self,
        tensor,
        *,
        round_idx,
        epoch,
        batch_idx,
        direction,
        client_id,
        model,
        stage,
    ):
        if not self.enabled:
            return tensor

        if tensor is None:
            return tensor

        packet_id = self._next_packet_id()

        cpu_tensor = tensor.detach().cpu().contiguous()

        serialized = self._serialize_tensor(cpu_tensor)

        payload_bytes = cpu_tensor.numel() * cpu_tensor.element_size()
        serialized_bytes = len(serialized)

        if cpu_tensor.numel() > 0:
            statistics_tensor = cpu_tensor.float()

            mean = statistics_tensor.mean().item()
            std = statistics_tensor.std().item()
            minimum = statistics_tensor.min().item()
            maximum = statistics_tensor.max().item()
        else:
            mean = 0.0
            std = 0.0
            minimum = 0.0
            maximum = 0.0

        record = PacketRecord(
            timestamp=time.time(),

            round=int(round_idx),
            epoch=int(epoch),
            batch=int(batch_idx),

            direction=direction,
            client_id=int(client_id),

            model=model,
            stage=stage,

            shape=list(cpu_tensor.shape),
            dtype=str(cpu_tensor.dtype),
            numel=int(cpu_tensor.numel()),

            payload_bytes=int(payload_bytes),
            serialized_bytes=int(serialized_bytes),

            mean=float(mean),
            std=float(std),
            min=float(minimum),
            max=float(maximum),
        )

        with self._lock:
            with open(self.jsonl_path, "a") as f:
                f.write(json.dumps(asdict(record)) + "\n")

        if self.save_payloads:
            payload_path = os.path.join(
                self.payload_dir,
                f"packet_{packet_id:08d}.bin"
            )

            with open(payload_path, "wb") as f:
                f.write(serialized)

        return tensor

    def summary(self):
        if not os.path.exists(self.jsonl_path):
            return {
                "packets": 0,
                "bytes": 0,
            }

        total_packets = 0
        total_bytes = 0

        with open(self.jsonl_path) as f:
            for line in f:
                record = json.loads(line)

                total_packets += 1
                total_bytes += record["serialized_bytes"]

        return {
            "packets": total_packets,
            "bytes": total_bytes,
        }
