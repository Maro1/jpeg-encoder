import numpy as np
import cv2
from scipy import fftpack
import huffman

BLOCK_SIZE = 8

y_quant_matrix = [16, 11, 10, 16, 24, 40, 51,
                  61, 12, 12, 14, 19, 26, 58,
                  60, 55, 14, 13, 16, 24, 40,
                  57, 69, 56, 14, 17, 22, 29,
                  51, 87, 80, 62, 18, 22, 37,
                  56, 68, 109, 103, 77, 24, 35,
                  55, 64, 81,  104, 113, 92,
                  49, 64, 78, 87, 103, 121, 120,
                  101, 72, 92, 95, 98, 112, 100,
                  103, 99]

c_quant_matrix = [17, 18, 24, 47, 99, 99, 99, 99,
                  18, 21, 26, 66, 99, 99, 99, 99,
                  24, 26, 56, 99, 99, 99, 99, 99,
                  47, 66, 99, 99, 99, 99, 99, 99,
                  99, 99, 99, 99, 99, 99, 99, 99,
                  99, 99, 99, 99, 99, 99, 99, 99,
                  99, 99, 99, 99, 99, 99, 99, 99,
                  99, 99, 99, 99, 99, 99, 99, 99]

zig_zag_order = [0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42,
                 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63]


def load_image(filepath: str) -> np.ndarray:
    return cv2.imread(filepath, cv2.IMREAD_COLOR)


def bgr_to_ycbcr(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)


def chroma_subsample(image: np.ndarray) -> tuple:
    height = image.shape[0]
    width = image.shape[1]

    Y = image[:, :, 0]
    CbCr = np.zeros((int(height / 2), int(width / 2), 2), dtype=np.uint8)

    for i in range(0, height, 2):
        for j in range(0, width, 2):
            cr = int(image[i, j, 1] / 4.0 + image[i + 1, j, 1] / 4.0 +
                     image[i, j + 1, 1] / 4.0 + image[i + 1, j + 1, 1] / 4.0)
            cb = int(image[i, j, 2] / 4.0 + image[i + 1, j, 2] / 4.0 +
                     image[i, j + 1, 2] / 4.0 + image[i + 1, j + 1, 2] / 4.0)
            CbCr[round(i/2), round(j/2)] = [cb, cr]

    return Y, CbCr


def make_blocks(image: np.ndarray) -> list:
    blocks_height = int(image.shape[0] / BLOCK_SIZE)
    blocks_width = int(image.shape[1] / BLOCK_SIZE)
    blocks = list()

    for i in range(blocks_height):
        for j in range(blocks_width):
            block = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=np.int)

            for block_row in range(BLOCK_SIZE):
                row = image[i * BLOCK_SIZE + block_row, j *
                            BLOCK_SIZE:j * BLOCK_SIZE + BLOCK_SIZE]
                block[block_row] = row

            blocks.append(block)

    return blocks


def run_dct(blocks: list) -> list:
    dct_blocks = list()
    for block in blocks:
        dct_block = fftpack.dct(block, axis=0, norm='ortho')
        dct_block = fftpack.dct(dct_block, axis=1, norm='ortho')
        dct_blocks.append(dct_block)

    return dct_blocks


def quantize(blocks: list, quant_matrix: list, scaling: int) -> list:
    quantized_blocks = list()
    quant_matrix = np.array(quant_matrix).reshape((BLOCK_SIZE, BLOCK_SIZE))

    for block in blocks:
        quantized = np.array([round(block[i, j] / round(quant_matrix[i, j] * scaling))
                              for i in range(BLOCK_SIZE) for j in range(BLOCK_SIZE)], dtype=np.int32)
        quantized_blocks.append(quantized)

    return quantized_blocks


def size_amp(components: list) -> list:
    transformed = list()
    for c in components:
        if c == 0:
            size = 0
        else:
            size = int(np.floor(np.log2(np.abs(c))) + 1)
        amplitude = c
        transformed.append((size, amplitude))
    return transformed


def run_length_code(blocks: list) -> list:
    rlc_blocks = list()
    for block in blocks:
        # Discard DC component
        block = block[1:]
        zero_counter = 0
        rlc = list()
        for i in range(len(block)):
            if not np.any(block[i:]):
                break
            num = block[i]
            if zero_counter == 16:
                rlc.append((15, 0))
                zero_counter = 0
            elif num == 0:
                zero_counter += 1
            else:
                rlc.append((zero_counter, num))
                zero_counter = 0
        rlc.append((0, 0))

        # Turn into ((RUNLENGTH, SIZE), AMPLITUDE) representation
        s_a = size_amp([x[1] for x in rlc])
        for i in range(len(rlc)):
            amplitude = s_a[i][1]
            runlength_size = (rlc[i][0], s_a[i][0])
            if (runlength_size == (0, 0)):
                rlc[i] = runlength_size
            else:
                rlc[i] = (runlength_size, amplitude)

        rlc_blocks.append(rlc)
    return rlc_blocks


def dpcm(dc_components: list) -> list:
    modulated = [dc_components[0]]

    for i in range(1, len(dc_components)):
        modulated.append(dc_components[i] - dc_components[i-1])

    return size_amp(modulated)


def ones_complement(num: int) -> str:
    if(num <= 0):  # if value==0, codeList = [], (SIZE,VALUE)=(SIZE)=EOB
        ret = list(bin(num)[3:])
        for i in range(len(ret)):
            if (ret[i] == '0'):
                ret[i] = 1
            else:
                ret[i] = 0
    else:
        ret = list(bin(num)[2:])
        for i in range(len(ret)):
            if (ret[i] == '0'):
                ret[i] = 0
            else:
                ret[i] = 1
    return ret


def write_to_file(filepath: str, dc: tuple, ac: tuple, image_size: tuple, scaling: int):
    with open(filepath, 'wb') as f:
        # Header
        f.write(bytes.fromhex('FFD8FFE000104A46494600010100000100010000'))

        # Luminance quantization table
        f.write(bytes.fromhex('FFDB004300'))
        quant_matrix = np.array(y_quant_matrix).reshape(
            (BLOCK_SIZE, BLOCK_SIZE))
        f.write(bytes(np.array([quant_matrix[i, j] * scaling for i in range(BLOCK_SIZE)
                for j in range(BLOCK_SIZE)], dtype=np.uint8)[zig_zag_order]))

        # Chrominance quantization table
        f.write(bytes.fromhex('FFDB004301'))
        quant_matrix = np.array(c_quant_matrix).reshape(
            (BLOCK_SIZE, BLOCK_SIZE))
        f.write(bytes(np.array([quant_matrix[i, j] * scaling for i in range(BLOCK_SIZE)
                for j in range(BLOCK_SIZE)], dtype=np.uint8)[zig_zag_order]))

        # Start of frame with image width/height
        f.write(bytes.fromhex('FFC0001108'))
        f.write(image_size[1].to_bytes(2, 'big'))
        f.write(image_size[0].to_bytes(2, 'big'))
        f.write(bytes.fromhex('03012200021101031101'))

        # Write Huffman tables
        f.write(bytes.fromhex(huffman.dc_lum_hex))
        f.write(bytes.fromhex(huffman.dc_chr_hex))
        f.write(bytes.fromhex(huffman.ac_lum_hex))
        f.write(bytes.fromhex(huffman.ac_chr_hex))

        # Start of scan
        f.write(bytes.fromhex('FFDA000C03010002110311003F00'))

        # Write image data
        y_dc_encoded = [[int(x) for x in list(huffman.dc_lum[d[0]]) + ones_complement(d[1])]
                        for d in dc[0]]
        cb_dc_encoded = [[int(x) for x in list(huffman.dc_chr[d[0]]) + ones_complement(d[1])]
                         for d in dc[1]]
        cr_dc_encoded = [[int(x) for x in list(huffman.dc_chr[d[0]]) + ones_complement(d[1])]
                         for d in dc[2]]

        y_ac_encoded = list()
        cb_ac_encoded = list()
        cr_ac_encoded = list()

        # Luminance
        for block in ac[0]:
            encoded_block = list()
            for el in block:
                if el == (0, 0):
                    y_encoded = [int(x) for x in list(
                        huffman.ac_lum[el])]
                else:
                    y_encoded = [int(x) for x in list(
                        huffman.ac_lum[el[0]])] + ones_complement(el[1])
                encoded_block.append(y_encoded)
            y_ac_encoded.append(encoded_block)

        # Cb
        for block in ac[1]:
            encoded_block = list()
            for el in block:
                if el == (0, 0):
                    cb_encoded = [int(x) for x in list(
                        huffman.ac_chr[el])]
                else:
                    cb_encoded = [int(x) for x in list(
                        huffman.ac_chr[el[0]])] + ones_complement(el[1])
                encoded_block.append(cb_encoded)
            cb_ac_encoded.append(encoded_block)

        # Cr
        for block in ac[2]:
            encoded_block = list()
            for el in block:
                if el == (0, 0):
                    cr_encoded = [int(x) for x in list(
                        huffman.ac_chr[el])]
                else:
                    cr_encoded = [int(x) for x in list(
                        huffman.ac_chr[el[0]])] + ones_complement(el[1])
                encoded_block.append(cr_encoded)
            cr_ac_encoded.append(encoded_block)

        # Write all encoded blocks
        encoded = list()
        for i in range(len(ac[0])):
            y = y_dc_encoded[i] + [x for y in y_ac_encoded[i] for x in y]

            encoded.extend(y)
            if (i + 1) % 4 == 0:
                j = int((i + 1) / 4) - 1
                cb = cb_dc_encoded[j] + \
                    [x for y in cb_ac_encoded[j] for x in y]
                cr = cr_dc_encoded[j] + \
                    [x for y in cr_ac_encoded[j] for x in y]

                encoded.extend(cb)
                encoded.extend(cr)

        encoded = ''.join([str(x) for x in encoded])

        # Pad 1 bits until byte-aligned
        while len(encoded) % 8 != 0:
            encoded += '1'

        encoded_bytes = bytearray(int(encoded, 2).to_bytes(
            len(encoded) // 8, byteorder='big'))

        # Non-marker 0xFF bytes need to be padded with 0x00
        padded_bytes = bytearray()
        for b in encoded_bytes:
            padded_bytes.append(b)
            if b == 255:
                padded_bytes.append(0)

        f.write(padded_bytes)

        # End of image
        f.write(bytes.fromhex('FFD9'))


def encode(filepath: str, q_factor: int) -> None:

    # Load image and convert to YCbCr color space
    image = load_image(filepath)
    image_size = (image.shape[1], image.shape[0])
    image = bgr_to_ycbcr(image)

    blocks_x = image.shape[1] // 8
    blocks_y = image.shape[0] // 8

    # Subsample the chroma channels in 4:2:0
    Y, CbCr = chroma_subsample(image)

    # Convert image channels to 8x8 blocks
    Y_blocks = make_blocks(Y)
    Cb_blocks = make_blocks(CbCr[:, :, 0])
    Cr_blocks = make_blocks(CbCr[:, :, 1])

    Y_blocks = [x - 128 for x in Y_blocks]
    Cb_blocks = [x - 128 for x in Cb_blocks]
    Cr_blocks = [x - 128 for x in Cr_blocks]

    # Run 2D DCT on all blocks
    Y_dct_blocks = run_dct(Y_blocks)
    Cb_dct_blocks = run_dct(Cb_blocks)
    Cr_dct_blocks = run_dct(Cr_blocks)

    if q_factor >= 50:
        scaling_factor = (100 - q_factor) / 50
    else:
        scaling_factor = 50 / q_factor

    # Quantize the DCT blocks assuming quality factor is less than 100
    if q_factor < 100:
        Y_dct_blocks = quantize(Y_dct_blocks, y_quant_matrix, scaling_factor)
        Cb_dct_blocks = quantize(Cb_dct_blocks, c_quant_matrix, scaling_factor)
        Cr_dct_blocks = quantize(Cr_dct_blocks, c_quant_matrix, scaling_factor)
    else:
        Y_dct_blocks = [x.astype(int) for x in Y_dct_blocks]
        Cb_dct_blocks = [x.astype(int) for x in Cb_dct_blocks]
        Cr_dct_blocks = [x.astype(int) for x in Cr_dct_blocks]

    # Reorder all blocks in zig-zag fashion
    Y_dct_blocks = [block.flatten()[zig_zag_order] for block in Y_dct_blocks]
    Cb_dct_blocks = [block.flatten()[zig_zag_order] for block in Cb_dct_blocks]
    Cr_dct_blocks = [block.flatten()[zig_zag_order] for block in Cr_dct_blocks]

    # Order Y blocks in 4x4 from top left to bottom right for interleaving
    Y_new = []
    for row in range(0, blocks_y, 2):
        for col in range(0, blocks_x, 2):
            Y_new.append(Y_dct_blocks[col + blocks_x*row])
            Y_new.append(Y_dct_blocks[col + blocks_x*row + 1])
            Y_new.append(Y_dct_blocks[col + blocks_x*(row + 1)])
            Y_new.append(Y_dct_blocks[col + blocks_x*(row + 1) + 1])
    Y_dct_blocks = Y_new

    Y_dc_components = [block[0] for block in Y_dct_blocks]
    Cb_dc_components = [block[0] for block in Cb_dct_blocks]
    Cr_dc_components = [block[0] for block in Cr_dct_blocks]

    Y_dpcm = dpcm(Y_dc_components)
    Cb_dpcm = dpcm(Cb_dc_components)
    Cr_dpcm = dpcm(Cr_dc_components)

    Y_rlc_blocks = run_length_code(Y_dct_blocks)
    Cb_rlc_blocks = run_length_code(Cb_dct_blocks)
    Cr_rlc_blocks = run_length_code(Cr_dct_blocks)

    dc = (Y_dpcm, Cb_dpcm, Cr_dpcm)
    ac = (Y_rlc_blocks, Cb_rlc_blocks, Cr_rlc_blocks)

    write_to_file('test.jpg', dc, ac, image_size, scaling_factor)

    return True


if __name__ == '__main__':
    encode('kodim23.png', 80)
