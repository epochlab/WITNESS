#!/usr/bin/env python3

def hex_chunk(chunk: bytes, start: int, end: int) -> str:
    return ' '.join(f'{b:02X}' for b in chunk[start:end])

def ascii_chunk(chunk: bytes, start: int, end: int) -> str:
    return ''.join(chr(b) if 32 <= b <= 127 else '.' for b in chunk[start:end])

def hexdump(chunk: bytes, length=256) -> None:
    for i in range(0, min(length, len(chunk)), 16):
        hex_part = hex_chunk(chunk, i, min(i+16, length))
        ascii_part = ascii_chunk(chunk, i, min(i+16, length))
        print(f"{i:08X}  {hex_part:<48}  {ascii_part}")