
## Image-Encryption-by-AES-Algorithm-upon-Image-Compression-by-DCT-and-Huffman-Coding

This project presents an integrated approach to secure image data management by combining image compression using Discrete Cosine Transform (DCT) and Huffman coding with image encryption using the Advanced Encryption Standard (AES) algorithm. The compressed image data is first optimized for storage efficiency, and then encrypted to ensure data confidentiality. This methodology aims to address both security and storage concerns in image data processing, offering a comprehensive solution for secure and efficient image data management

![Workflow Image](https://raw.githubusercontent.com/ChitrakshGupta/Image-Encryption-by-AES-Algorithm-upon-Image-Compression-by-DCT-and-Huffman-Coding/master/Workflow.jpg)

The implemented project achieved significant reductions in image sizes through compression using Discrete Cosine Transform (DCT) and Huffman coding, with compression ratios averaging X . Evaluation of image quality metrics such as peak signal-to-noise ratio (PSNR) and structural similarity index (SSIM) indicated minimal loss of visual fidelity post-compression. The AES encryption algorithm effectively secured the compressed images, demonstrating robust encryption strength and negligible computational overhead. Integration of compression and encryption techniques resulted in an overall improvement in image security while maintaining optimal storage and transmission efficiency, as evidenced by reduced file sizes and enhanced data protection. Performance metrics including encryption/decryption speeds and memory usage were within acceptable limits, highlighting the viability of the dual-layered approach in enhancing image data security and efficiency.



![Project Overview Image](https://raw.githubusercontent.com/ChitrakshGupta/Image-Encryption-by-AES-Algorithm-upon-Image-Compression-by-DCT-and-Huffman-Coding/master/Project%20Overview.jpg)






## Installation

1. Clone the repository using Git.
2. First, run `encrypApp.py`.
   - Provide the input image.
   - The encrypted image, key, and codeword table will be shown in the `cipher` folder.
3. Then, run `decryptApp.py`.
4. After running it, you will see the output of the final image in the `output` folder.


## Tech Stack

- **Programming Languages:** Python
- **Libraries/Frameworks:**
  - PyCryptoDome (for AES encryption)
  - NumPy (for handling arrays and mathematical operations, possibly used in DCT implementation)
- Cryptography
- Image Processing


##  Reference

- \item Image DCT: {https://www.math.cuhk.edu.hk/~lmlui/dct.pdf}
- \item Image Huffman: {https://www.nayuki.io/page/reference-huffman-coding}
- \item AES: {https://blog.nindalf.com/posts/implementing-aes/}

For details, please refer to [Report.pdf](https://github.com/ChitrakshGupta/Image-Encryption-by-AES-Algorithm-upon-Image-Compression-by-DCT-and-Huffman-Coding/blob/master/Project.pdf)

For further information or inquiries, please contact:

**Chitraksh Gupta**  
Email: [guptachitraksh@gmail.com](mailto:guptachitraksh@gmail.com)  

