import torch
import numpy as np
import random
from swin1d.module import swin_1d_block
from swin1d.examples import generate_random_dna, onehot_encoder
import matplotlib.pyplot as plt

#Set random seeds for reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

def visualize_dna_sequence(sequence):
    if isinstance(sequence, list):
        # Concatenate the list of strings into a single string
        sequence = ''.join(sequence)
    mapping = {'A': 0, 'T': 0.5, 'C': 1, 'G': 1.5}
    colors = np.array([mapping[base] for base in sequence])

    plt.imshow(colors.reshape(1, -1), cmap='coolwarm', aspect='auto')
    plt.xticks([])
    plt.yticks([])
    plt.title("DNA Sequence")
    plt.show()

# def visualize_dna_sequence(sequence):
#     mapping = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
#     colors = np.array([mapping.get(base,0) for base in sequence])
#     max_color_value = max(colors)
#     normalized_colors = colors / max_color_value if max_color_value > 0 else colors

#     plt.imshow(normalized_colors.reshape(1, -1), cmap='viridis', aspect='auto')
#     plt.xticks([])
#     plt.yticks([])
#     plt.title("DNA Sequence Visualization")
#     plt.show()

def extract_white_part(image_array, threshold=0.5):
    white_part = (image_array > threshold).astype(float)
    return white_part

def test_genomic_model(seq_length=512):
    #input_sequence = generate_random_dna(seq_length)

    input_sequence=['GGCTGGCAGCATGACCGCACCAGGGGGGCAAATCATGGTACGCGTCTCCGGAGCGGGGGGGTATCGCCTGGCGGCCGTGACTGTTGAGAAGAGCGTGGTACGGGATAGACGGCCCTGCCCAGGAGCTACACGTGCTGTCTCTATAGCCTGTCTACGGGGGTAGCCCTGAAGCTAGGTGGAGTCAGACGCATCGCTCCGGGTCGCGACAAGTGAGACACCATAACTTCGTTATAGCCCGGCGCTCCATGCATTCCGCCTTAGCAGACACAAGAACGGAGGGGGATCAGATGTACGTCCGAAGTGGCTTACGTACAGATATTGAAGTATTATGTGGTTCGAGCTGTTCTATAGGTACTACTCCTATAAGATGTGCGGGGACCCGGGAGTCATGCGCAGGAATTCGCGTCCTCTTGGTGGTCCGACCGTGTCACGTTTATCTGACGCTGGGAAGCCTGGGGAGTTATCTGAGGGGTAAAGATACCACGGGCCGGGGCGACGGACATCGAAGCAGA']
    print("Generated DNA Sequence:", input_sequence)
    encode_input = onehot_encoder(input_sequence)
    model = swin1d_block(4)
    model.eval() 
    output = model(encode_input)
    print(output.shape)

    output_array = output.detach().numpy().squeeze()
    plt.subplot(1, 3, 1)
    plt.imshow(output_array, cmap='gray')
    plt.title("Genomic Model Output")

    white_part = extract_white_part(output_array)
    plt.subplot(1, 3, 2)
    plt.imshow(output_array, cmap='gray')
    plt.title("Original Image")

    plt.subplot(1, 3, 3)
    plt.imshow(white_part, cmap='gray')
    plt.title("White Part")
    plt.show()

    visualize_dna_sequence(input_sequence)

def swin1d_block(dim):
    window_size = 32
    stages = [
        (4, True, window_size),
        (2, False, window_size),
        (2, False, window_size),
        (2, False, window_size),
    ]
    model = swin_1d_block(stages, dim)
    return model

if __name__ == "__main__":
    test_genomic_model()
