# ---------- compact.py ----------

import pydicom
import numpy as np
import os
import zlib
from hilbertcurve.hilbertcurve import HilbertCurve
from imageio import imwrite
import cv2

# ---------- Hilbert Scan ---------- 
def hilbert_ordering(shape):
    height, width = shape
    max_dim = max(height, width)
    n = int(np.ceil(np.log2(max_dim)))
    hilbert_curve = HilbertCurve(p=n, n=2)

    coords = []
    for i in range(2**n):
        for j in range(2**n):
            if i < height and j < width:
                d = hilbert_curve.distance_from_point([i, j])
                coords.append((d, i, j))

    coords.sort()
    return [(i, j) for _, i, j in coords]

def apply_hilbert(image):
    order = hilbert_ordering(image.shape)
    return image[[i for i, _ in order], [j for _, j in order]]

def reverse_hilbert(flat_pixels, shape):
    order = hilbert_ordering(shape)
    image = np.zeros(shape, dtype=np.uint16)
    for val, (i, j) in zip(flat_pixels, order):
        image[i, j] = val
    return image

# ---------- Delta Encoding ----------
def compute_delta(a, b):
    return int(b) - int(a)

def is_large_delta(delta):
    return delta <= -64 or delta >= 65

def delta_encode(pixels):
    encoded = []
    prev = pixels[0]
    encoded.append(('first', prev))

    for i in range(1, len(pixels)):
        delta = compute_delta(prev, pixels[i])
        if is_large_delta(delta):
            encoded.append(('full', delta))
        else:
            encoded.append(('short', delta))
        prev = pixels[i]

    return encoded

def delta_decode(encoded_stream):
    pixels = []
    for tag, val in encoded_stream:
        if tag == 'first':
            pixels.append(val)
            prev = val
        else:
            delta = val
            current = prev + delta
            pixels.append(current)
            prev = current
    return pixels

# ---------- Byte Encoding ----------
def encode_to_bytes(encoded_stream):
    out = bytearray()
    for tag, val in encoded_stream:
        if tag == 'first':
            out += int(val).to_bytes(2, 'big')
        elif tag == 'short':
            val_shifted = val + 64
            if 0 <= val_shifted <= 255:
                out += val_shifted.to_bytes(1, 'big')
            else:
                out += b'\xFF' + int(val).to_bytes(2, 'big', signed=True)
        elif tag == 'full':
            out += b'\xFF' + int(val).to_bytes(2, 'big', signed=True)
    return out

def decode_from_bytes(byte_data):
    i = 0
    stream = []
    while i < len(byte_data):
        if i == 0:
            first = int.from_bytes(byte_data[0:2], 'big')
            stream.append(('first', first))
            i = 2
        elif byte_data[i] == 0xFF:
            delta = int.from_bytes(byte_data[i+1:i+3], 'big', signed=True)
            stream.append(('full', delta))
            i += 3
        else:
            delta = byte_data[i] - 64
            stream.append(('short', delta))
            i += 1
    return stream

# ---------- Normalization & Enhancement ----------
def normalize_image(image):
    image = image.astype(np.float32)
    image -= image.min()
    if image.max() > 0:
        image /= image.max()
    image *= 255.0
    return image.astype(np.uint8)

# ---------- Compression / Decompression ----------
def compact_compress(dicom_path, output_path):
    ds = pydicom.dcmread(dicom_path)
    image = ds.pixel_array.astype(np.uint16)
    shape = image.shape
    h_image = apply_hilbert(image).flatten()

    encoded = delta_encode(h_image)
    byte_stream = encode_to_bytes(encoded)
    compressed = zlib.compress(byte_stream)

    with open(output_path, 'wb') as f:
        f.write(np.array(shape, dtype=np.uint16).tobytes())
        f.write(compressed)

def compact_decompress(compressed_path, output_path):
    with open(compressed_path, 'rb') as f:
        raw = f.read()

    shape = tuple(np.frombuffer(raw[:4], dtype=np.uint16))
    compressed_data = raw[4:]
    byte_stream = zlib.decompress(compressed_data)

    encoded_stream = decode_from_bytes(byte_stream)
    flat_pixels = delta_decode(encoded_stream)

    recovered_image = reverse_hilbert(flat_pixels, shape)

    # Normalize only (no false color enhancement)
    grayscale = normalize_image(recovered_image)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imwrite(output_path, grayscale)


