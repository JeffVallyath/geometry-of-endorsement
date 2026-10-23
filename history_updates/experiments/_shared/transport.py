"""Lossless, streaming transport for recorded gzip payloads.

XZ stores the exact decoded bytes. Compatibility paths use a deterministic gzip
envelope with no collection date or original filename in its header.
"""
import hashlib
import lzma
import struct
import zlib

HEADER=bytes.fromhex('1f8b08000000000002ff')


def gzip_chunks(chunks):
    yield HEADER
    compressor=zlib.compressobj(9,zlib.DEFLATED,-zlib.MAX_WBITS)
    crc=0;size=0
    for chunk in chunks:
        size+=len(chunk);crc=zlib.crc32(chunk,crc)
        compressed=compressor.compress(chunk)
        if compressed:yield compressed
    yield compressor.flush()
    yield struct.pack('<II',crc,size%(1<<32))


def chunks(stream):
    while chunk:=stream.read(1<<20):yield chunk


def restore(source,target,record):
    if record['codec']!='xz-to-gzip-v1':raise ValueError('Unknown transport codec')
    measured=hashlib.sha256();size=0;decoded=hashlib.sha256();decoded_size=0
    def observed(stream):
        nonlocal decoded_size
        for chunk in chunks(stream):
            decoded.update(chunk);decoded_size+=len(chunk);yield chunk
    with lzma.open(source,'rb') as compressed,target.open('xb') as destination:
        for chunk in gzip_chunks(observed(compressed)):
            measured.update(chunk);size+=len(chunk);destination.write(chunk)
    if (measured.hexdigest()!=record['materialized_sha256'] or size!=record['materialized_bytes']
        or decoded.hexdigest()!=record['decoded_sha256'] or decoded_size!=record['decoded_bytes']):
        raise ValueError('Transport reconstruction mismatch')
