"""
This code implements two functions for zigzag scanning and inverse zigzag scanning of a matrix that used in DCT implementation.
"""
import numpy as np
import cv2
import math
from Crypto.Cipher import AES

def zigzag(input):
    h = 0
    v = 0

    vmin = 0
    hmin = 0

    vmax = input.shape[0]
    hmax = input.shape[1]

    i = 0

    
    output = np.zeros((vmax * hmax))

    while ((v < vmax) and (h < hmax)):
        if ((h + v) % 2) == 0:
            if (v == vmin):
                output[i] = input[v, h]
                if (h == hmax):
                    v = v + 1
                else:
                    h = h + 1
                i = i + 1
            elif ((h == hmax - 1) and (v < vmax)):
                output[i] = input[v, h]
                v = v + 1
                i = i + 1
            elif ((v > vmin) and (h < hmax - 1)):
                output[i] = input[v, h]
                v = v - 1
                h = h + 1
                i = i + 1
        else:
            if ((v == vmax - 1) and (h <= hmax - 1)):
                output[i] = input[v, h]
                h = h + 1
                i = i + 1
            elif (h == hmin):
                output[i] = input[v, h]
                if (v == vmax - 1):
                    h = h + 1
                else:
                    v = v + 1
                i = i + 1
            elif ((v < vmax - 1) and (h > hmin)):
                output[i] = input[v, h]
                v = v + 1
                h = h - 1
                i = i + 1

        if ((v == vmax - 1) and (h == hmax - 1)):
            output[i] = input[v, h]
            break

    return output
"""
This code performs image compression using Discrete Cosine Transform (DCT) and run-length encoding.

0000000"""


def get_run_length_encoding(image):
    i = 0
    skip = 0
    stream = []    
    bitstream = ""
    image = image.astype(int)
    while i < image.shape[0]:
        if image[i] != 0:            
            stream.append((image[i],skip))
            bitstream = bitstream + str(image[i])+ " " +str(skip)+ " "
            skip = 0
        else:
            skip = skip + 1
        i = i + 1

    return bitstream

block_size = 8

QUANTIZATION_MAT = np.array([[16,11,10,16,24,40,51,61],[12,12,14,19,26,58,60,55],[14,13,16,24,40,57,69,56 ],[14,17,22,29,51,87,80,62],[18,22,37,56,68,109,103,77],[24,35,55,64,81,104,113,92],[49,64,78,87,103,121,120,101],[72,92,95,98,112,100,103,99]])

img = cv2.imread('Input/img.jpg', cv2.IMREAD_GRAYSCALE)

[h , w] = img.shape

height = h
width = w
h = np.float32(h) 
w = np.float32(w) 

nbh = math.ceil(h/block_size)
nbh = np.int32(nbh)

nbw = math.ceil(w/block_size)
nbw = np.int32(nbw)

H =  block_size * nbh

W =  block_size * nbw

padded_img = np.zeros((H,W))

padded_img[0:height,0:width] = img[0:height,0:width]

cv2.imwrite('uncompressed.bmp', np.uint8(padded_img))

for i in range(nbh):
        row_ind_1 = i*block_size                
        row_ind_2 = row_ind_1+block_size
        
        for j in range(nbw):
            col_ind_1 = j*block_size                       
            col_ind_2 = col_ind_1+block_size
                        
            block = padded_img[ row_ind_1 : row_ind_2 , col_ind_1 : col_ind_2 ]          
            DCT = cv2.dct(block)            
            DCT_normalized = np.divide(DCT,QUANTIZATION_MAT).astype(int)            
            reordered = zigzag(DCT_normalized)
            reshaped= np.reshape(reordered, (block_size, block_size)) 
            padded_img[row_ind_1 : row_ind_2 , col_ind_1 : col_ind_2] = reshaped                        

cv2.imwrite('compressed_image.bmp', np.uint8(padded_img))

arranged = padded_img.flatten()
bitstream = get_run_length_encoding(arranged)
bitstream = str(padded_img.shape[0]) + " " + str(padded_img.shape[1]) + " " + bitstream + ";"
file1 = open("image2.txt","w")
file1.write(bitstream)
file1.close()
"""this block of code removing last semicolan in the text file"""
def remove_last_character(input_file):
    with open(input_file, 'r') as file:
        text = file.read()
        if text:
            modified_text = text[:-1]  # Remove the last character
            with open(input_file, 'w') as file:
                file.write(modified_text)
            print(f"Last character removed from {input_file} and saved successfully.")
        else:
            print("File is empty.")

# Usage example
input_file = 'image2.txt'  # Replace 'example.txt' with your file name
remove_last_character(input_file)

"""
Uses heapq for priority queue operations in Huffman compression.
Implements Huffman encoding with heapq, Counter, and defaultdict.
"""

import heapq

from collections import Counter, defaultdict

class HuffmanNode:
    def __init__(self, symbol, freq):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_frequency_table(data):
    freq_table = Counter(data)
    return freq_table

def build_huffman_tree(freq_table):
    priority_queue = [HuffmanNode(symbol, freq) for symbol, freq in freq_table.items()]
    heapq.heapify(priority_queue)
    
    while len(priority_queue) > 1:
        left = heapq.heappop(priority_queue)
        right = heapq.heappop(priority_queue)
        merged_node = HuffmanNode(None, left.freq + right.freq)
        merged_node.left = left
        merged_node.right = right
        heapq.heappush(priority_queue, merged_node)

    return priority_queue[0]

def build_codeword_table(node, current_code="", codeword_table=None):
    if codeword_table is None:
        codeword_table = {}
    if node.symbol is not None:
        codeword_table[node.symbol] = current_code
    if node.left is not None:
        build_codeword_table(node.left, current_code + "0", codeword_table)
    if node.right is not None:
        build_codeword_table(node.right, current_code + "1", codeword_table)
    return codeword_table

def huffman_compress(data):
    freq_table = build_frequency_table(data)
    huffman_tree = build_huffman_tree(freq_table)
    codeword_table = build_codeword_table(huffman_tree)
    compressed_data = "".join(codeword_table[symbol] for symbol in data)
    return compressed_data, codeword_table

def huffman_decompress(compressed_data, codeword_table):
    reverse_codeword_table = {code: symbol for symbol, code in codeword_table.items()}
    decompressed_data = []
    current_code = ""
    for bit in compressed_data:
        current_code += bit
        if current_code in reverse_codeword_table:
            decompressed_data.append(reverse_codeword_table[current_code])
            current_code = ""
    return decompressed_data

# Read data from image2.txt
with open('image2.txt', 'r') as file:
    data = file.read().split()

# Compress data
compressed_data, codeword_table = huffman_compress(data)

# Save compressed data to a file
with open('compressed_data.txt', 'w') as file:
    file.write(compressed_data)

# Decompress data from the compressed file
with open('compressed_data.txt', 'r') as file:
    compressed_data = file.read()

decompressed_data = huffman_decompress(compressed_data, codeword_table)

# Save codeword table to a file
with open('Cipher/codeword_table.txt', 'w') as file:
    for symbol, code in codeword_table.items():
        file.write(f"{symbol}: {code}\n")

print("Codeword table saved to codeword_table.txt.")

# # Save decompressed data to a file
# with open('decompressed_data.txt', 'w') as file:
#     file.write(' '.join(decompressed_data))

print("Compression  completed.")
from Crypto.Cipher import AES
import os
import os
import os

# Generate a 16-byte (128-bit) key
key = os.urandom(16)

# Save the key to a text file
with open('encryption_key.txt', 'wb') as file:
    file.write(key)

print("Key saved to encryption_key.txt.")



def encrypt_file(input_file, output_file, key):
    chunk_size = 64 * 1024  # 64 KB
    cipher = AES.new(key, AES.MODE_ECB)
    with open(input_file, 'rb') as f_input, open(output_file, 'wb') as f_output:
        while True:
            chunk = f_input.read(chunk_size)
            if len(chunk) == 0:
                break
            elif len(chunk) % 16 != 0:
                chunk += b' ' * (16 - len(chunk) % 16)  # Padding
            encrypted_chunk = cipher.encrypt(chunk)
            f_output.write(encrypted_chunk)
input_file = 'compressed_data.txt'
encrypted_file = 'Cipher/encrypted_file.bin'
encrypt_file(input_file, encrypted_file, key)

