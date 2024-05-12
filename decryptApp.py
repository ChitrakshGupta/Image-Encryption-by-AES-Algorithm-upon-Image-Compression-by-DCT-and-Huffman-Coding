from Crypto.Cipher import AES
import cv2
import numpy as np
import math

def inverse_zigzag(input, vmax, hmax):
    h = 0
    v = 0

    vmin = 0
    hmin = 0

    output = np.zeros((vmax, hmax))

    i = 0

    while ((v < vmax) and (h < hmax)):
        if ((h + v) % 2) == 0:
            if (v == vmin):
                output[v, h] = input[i]
                if (h == hmax):
                    v = v + 1
                else:
                    h = h + 1
                i = i + 1
            elif ((h == hmax - 1) and (v < vmax)):
                output[v, h] = input[i]
                v = v + 1
                i = i + 1
            elif ((v > vmin) and (h < hmax - 1)):
                output[v, h] = input[i]
                v = v - 1
                h = h + 1
                i = i + 1
        else:
            if ((v == vmax - 1) and (h <= hmax - 1)):
                output[v, h] = input[i]
                h = h + 1
                i = i + 1
            elif (h == hmin):
                output[v, h] = input[i]
                if (v == vmax - 1):
                    h = h + 1
                else:
                    v = v + 1
                i = i + 1
            elif ((v < vmax - 1) and (h > hmin)):
                output[v, h] = input[i]
                v = v + 1
                h = h - 1
                i = i + 1

        if ((v == vmax - 1) and (h == hmax - 1)):
            output[v, h] = input[i]
            break

    return output



def read_key_from_file(key_file):
    with open(key_file, 'rb') as file:
        key = file.read()
    return key

def decrypt_file(input_file, output_file, key):
    chunk_size = 64 * 1024  # 64 KB
    cipher = AES.new(key, AES.MODE_ECB)
    with open(input_file, 'rb') as f_input, open(output_file, 'wb') as f_output:
        while True:
            chunk = f_input.read(chunk_size)
            if len(chunk) == 0:
                break
            decrypted_chunk = cipher.decrypt(chunk)
            f_output.write(decrypted_chunk.rstrip(b' '))  # Remove padding

# Specify the paths to the encrypted file and the decrypted output file
encrypted_file = 'Cipher/encrypted_file.bin'
decrypted_file = 'decrypted_data.txt'
key_file = 'Cipher/encryption_key.txt'

# Read the key from the key file
key = read_key_from_file(key_file)

# Decrypt the file using the key
decrypt_file(encrypted_file, decrypted_file, key)


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

# Load codeword table from file
codeword_table = {}
with open('Cipher/codeword_table.txt', 'r') as file:
    for line in file:
        symbol, code = line.strip().split(': ')
        codeword_table[symbol] = code

# Read compressed data from file
compressed_file_path = 'compressed_data.txt'
with open(compressed_file_path, 'r') as file:
    compressed_data = file.read()

# Decompress data
decompressed_data = huffman_decompress(compressed_data, codeword_table)

# Save decompressed data to a file
with open('decompressed_data.txt', 'w') as file:
    file.write(' '.join(decompressed_data))

print("Decompression completed.")

"""Adding semicolan to the last of text file for inverse idct"""

def add_semicolon_to_file(file_path):
    # Read the content of the file
    with open(file_path, 'r') as file:
        content = file.read()
    content += ' ;'
    with open(file_path, 'w') as file:
        file.write(content)

# Example usage
file_path = 'decompressed_data.txt'
add_semicolon_to_file(file_path)
print(f"Semicolon added to {file_path}")

"""Inverse-DCT converting back into the image"""
import cv2
import numpy as np
import math
QUANTIZATION_MAT = np.array([[16,11,10,16,24,40,51,61],[12,12,14,19,26,58,60,55],[14,13,16,24,40,57,69,56 ],[14,17,22,29,51,87,80,62],[18,22,37,56,68,109,103,77],[24,35,55,64,81,104,113,92],[49,64,78,87,103,121,120,101],[72,92,95,98,112,100,103,99]])
block_size = 8
with open('decompressed_data.txt', 'r') as myfile:
    image=myfile.read()
details = image.split()
h = int(''.join(filter(str.isdigit, details[0])))
w = int(''.join(filter(str.isdigit, details[1])))
array = np.zeros(h*w).astype(int)
k = 0
i = 2
x = 0
j = 0

while k < array.shape[0]:
# Oh! image has ended
    if(details[i] == ';'):
        break
    if "-" not in details[i]:
        array[k] = int(''.join(filter(str.isdigit, details[i])))        
    else:
        array[k] = -1*int(''.join(filter(str.isdigit, details[i])))        

    if(i+3 < len(details)):
        j = int(''.join(filter(str.isdigit, details[i+3])))

    if j == 0:
        k = k + 1
    else:                
        k = k + j + 1        

    i = i + 2

array = np.reshape(array,(h,w))
i = 0
j = 0
k = 0
padded_img = np.zeros((h,w))

while i < h:
    j = 0
    while j < w:        
        temp_stream = array[i:i+8,j:j+8]                
        block = inverse_zigzag(temp_stream.flatten(), int(block_size),int(block_size))            
        de_quantized = np.multiply(block,QUANTIZATION_MAT)                
        padded_img[i:i+8,j:j+8] = cv2.idct(de_quantized)        
        j = j + 8        
    i = i + 8
padded_img[padded_img > 255] = 255
padded_img[padded_img < 0] = 0
cv2.imwrite("final_image.bmp",np.uint8(padded_img))

# DONE!