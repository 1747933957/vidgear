
# -*- coding: utf-8 -*-
"""
packets.py
----------
三类包定义（DataPacket / FecPacket / XorPacket）以及编解码。
- 外层统一包头：MAGIC(2)+VER(1)+TYPE(1)。
- XOR 内层项：frameId(U32) + len32(U32) + credit(U16)（每项 10B）。
- FEC：包含 parity（覆盖分片 payload 的逐字节 XOR）。
  编码：MAGIC,VER,TYPE,frame_id(U32),count(U16), [ids...](U16*count), parity_len(U16), parity。
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, ClassVar
import struct

MAGIC = b"SP"
VER   = 1
T_DATA, T_FEC, T_XOR = 1, 2, 3

XOR_MAGIC3 = b"XOR"
XOR_VER    = 1

def _pack(fmt: str, *vals) -> bytes:
    return struct.pack(fmt, *vals)

def _unpack(fmt: str, data: bytes, offset: int = 0):
    size = struct.calcsize(fmt)
    return struct.unpack(fmt, data[offset: offset+size]), offset+size

@dataclass
class DataPacket:
    frame_id: int
    frame_len: int
    frame_pkt_num: int
    pkt_id_in_frame: int
    payload: bytes
    HDR_FMT: ClassVar[str] = "!2sBBI I H H H"
    HDR_SIZE: ClassVar[int] = struct.calcsize(HDR_FMT)
    def to_bytes(self) -> bytes:
        pay_len = len(self.payload)
        hdr = _pack(self.HDR_FMT, MAGIC, VER, T_DATA, self.frame_id, self.frame_len,
                    self.frame_pkt_num, self.pkt_id_in_frame, pay_len)
        return hdr + self.payload
    @classmethod
    def from_bytes(cls, data: bytes) -> "DataPacket":
        (magic, ver, typ, fid, flen, n, pid, plen), p = _unpack(cls.HDR_FMT, data, 0)
        assert magic == MAGIC and ver == VER and typ == T_DATA, "Not a DataPacket"
        payload = data[p:p+plen]
        return cls(fid, flen, n, pid, payload)

@dataclass
class FecPacket:
    frame_id: int
    covered_ids: List[int]
    parity: bytes
    HDR_FMT: ClassVar[str] = "!2sBBI H"
    HDR_SIZE: ClassVar[int] = struct.calcsize(HDR_FMT)
    def to_bytes(self) -> bytes:
        cnt = len(self.covered_ids)
        hdr = _pack(self.HDR_FMT, MAGIC, VER, T_FEC, self.frame_id, cnt)
        ids = b"".join(_pack('!H', i) for i in self.covered_ids)
        par_len = _pack('!H', len(self.parity))
        return hdr + ids + par_len + self.parity
    @classmethod
    def from_bytes(cls, data: bytes) -> "FecPacket":
        (magic, ver, typ, fid, cnt), p = _unpack(cls.HDR_FMT, data, 0)
        assert magic == MAGIC and ver == VER and typ == T_FEC, "Not a FecPacket"
        covered = []
        for _ in range(cnt):
            (i,), p = _unpack("!H", data, p); covered.append(i)
        (plen,), p = _unpack("!H", data, p)
        parity = data[p:p+plen]
        return cls(fid, covered, parity)

@dataclass
class XorPacket:
    anchor_frame_id: int
    items: List[Tuple[int, int, int]]  # (frameId, fullLen32, credit16)
    budget: int
    OUTER_FMT: ClassVar[str] = "!2sBBI I"
    OUTER_SIZE: ClassVar[int] = struct.calcsize(OUTER_FMT)
    def to_bytes(self) -> bytes:
        meta_overhead = 8
        item_size     = 10
        need = meta_overhead + len(self.items) * item_size
        if self.budget < need:
            raise ValueError(f"XorPacket: items metadata length exceeds budget (need={need}, budget={self.budget})")
        outer = _pack(self.OUTER_FMT, MAGIC, VER, T_XOR, self.anchor_frame_id, self.budget)
        inner = bytearray()
        inner += XOR_MAGIC3
        inner += bytes([XOR_VER])
        inner += _pack("!H", 0)
        inner += _pack("!H", len(self.items))
        for (fid, length32, credit16) in self.items:
            inner += _pack("!IIH", fid, length32, credit16)
        inner += b"\x00" * (self.budget - need)
        return outer + bytes(inner)
    @classmethod
    def from_bytes(cls, data: bytes) -> "XorPacket":
        (magic, ver, typ, anchor, budget), p = _unpack(cls.OUTER_FMT, data, 0)
        assert magic == MAGIC and ver == VER and typ == T_XOR, "Not a XorPacket"
        inner = data[p:p+budget]
        if len(inner) < 4:
            return cls(anchor, [], budget)
        if inner[:3] != XOR_MAGIC3 or inner[3] != XOR_VER:
            return cls(anchor, [], budget)
        off  = struct.unpack("!H", inner[4:6])[0]
        cnt  = struct.unpack("!H", inner[6:8])[0]
        meta_overhead = 8
        item_size     = 10
        need = meta_overhead + cnt * item_size
        if len(inner) < need:
            raise ValueError("Corrupted XOR payload: not enough bytes for items")
        items: List[Tuple[int,int,int]] = []
        pos = 8
        for _ in range(cnt):
            fid, length32, credit16 = struct.unpack("!IIH", inner[pos:pos+10])
            items.append((fid, length32, credit16))
            pos += 10
        return cls(anchor, items, budget)
