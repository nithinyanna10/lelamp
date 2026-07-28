from __future__ import annotations

# Feetech SCS/STS half-duplex protocol: 0xFF 0xFF <id> <len> <instr> <params...> <checksum>.
# ponytail: register addresses follow the commonly published STS3215 memory table
# (as used by lerobot's feetech driver). Verify against your servo firmware before
# flashing to real hardware -- there's no bench here to confirm byte-for-byte.
INSTR_WRITE = 0x03
INSTR_SYNC_WRITE = 0x83

ADDR_GOAL_POSITION = 42  # 2 bytes
ADDR_PRESENT_POSITION = 56  # 2 bytes

_POSITION_RANGE_COUNTS = 4096  # 12-bit magnetic encoder, 0..4095 over 360 degrees


def _checksum(body: bytes) -> int:
    return (~sum(body)) & 0xFF


def angle_to_raw(angle_rad: float, joint_range: tuple[float, float]) -> int:
    lo, hi = joint_range
    frac = (angle_rad - lo) / (hi - lo) if hi > lo else 0.0
    frac = min(max(frac, 0.0), 1.0)
    return round(frac * (_POSITION_RANGE_COUNTS - 1))


def raw_to_angle(raw: int, joint_range: tuple[float, float]) -> float:
    lo, hi = joint_range
    frac = raw / (_POSITION_RANGE_COUNTS - 1)
    return lo + frac * (hi - lo)


def build_sync_write_goal_position(servo_ids: list[int], raw_positions: list[int]) -> bytes:
    """One packet setting Goal_Position for every servo in a single bus transaction."""
    params = bytearray([ADDR_GOAL_POSITION, 2])
    for servo_id, raw in zip(servo_ids, raw_positions, strict=True):
        params += bytes([servo_id, raw & 0xFF, (raw >> 8) & 0xFF])
    length = len(params) + 2
    body = bytes([0xFE, length, INSTR_SYNC_WRITE]) + bytes(params)
    return bytes([0xFF, 0xFF]) + body + bytes([_checksum(body)])
