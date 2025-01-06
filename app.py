from flask import Flask, render_template, request
import torch
import numpy as np
from swin1d.module import swin_1d_block
from swin1d.examples import onehot_encoder
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

def visualize_dna_sequence(sequence):
    if isinstance(sequence, list):
        sequence = ''.join(sequence)
    mapping = {'A': 0, 'T': 0.5, 'C': 1, 'G': 1.5}
    colors = np.array([mapping[base] for base in sequence])

    plt.imshow(colors.reshape(1, -1), cmap='coolwarm', aspect='auto')
    plt.xticks([])
    plt.yticks([])
    plt.title("DNA Sequence")

    # Convert plot to base64 encoded string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return plot_data

def extract_white_part(image_array, threshold=0.5):
    white_part = (image_array > threshold).astype(float)
    return white_part

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

def is_valid_sequence(sequence, target_length=512):
    return len(sequence) == target_length

@app.route('/', methods=['GET', 'POST'])
def index():
    error_message = None
    plot_data = None
    dna_plot = None

    if request.method == 'POST':
        input_sequence = request.form['sequence']

        if not is_valid_sequence(input_sequence):
            error_message = f"The sequence length should be {512}. Please enter a valid sequence."

        else:
            encode_input = onehot_encoder([input_sequence])
            model = swin1d_block(4)
            model.eval()
            output = model(encode_input)
            output_array = output.detach().numpy().squeeze()

            plt.figure(figsize=(12, 4))

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

            # Convert the plots to base64 encoded strings
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            plt.close()
            plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

            dna_plot = visualize_dna_sequence(input_sequence)

    return render_template('index.html', error_message=error_message, plot_data=plot_data, dna_plot=dna_plot)


if __name__ == "__main__":
    app.run(debug=True)


# from flask import Flask, render_template, request
# import torch
# import numpy as np
# from swin1d.module import swin_1d_block
# from swin1d.examples import onehot_encoder
# import matplotlib.pyplot as plt

# app = Flask(__name__)

# def visualize_dna_sequence(sequence):
#     if isinstance(sequence, list):
#         sequence = ''.join(sequence)
#     mapping = {'A': 0, 'T': 0.5, 'C': 1, 'G': 1.5}
#     colors = np.array([mapping[base] for base in sequence])

#     plt.imshow(colors.reshape(1, -1), cmap='coolwarm', aspect='auto')
#     plt.xticks([])
#     plt.yticks([])
#     plt.title("DNA Sequence")
#     plt.show()

# def extract_white_part(image_array, threshold=0.5):
#     white_part = (image_array > threshold).astype(float)
#     return white_part

# def swin1d_block(dim):
#     window_size = 32
#     stages = [
#         (4, True, window_size),
#         (2, False, window_size),
#         (2, False, window_size),
#         (2, False, window_size),
#     ]
#     model = swin_1d_block(stages, dim)
#     return model

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         input_sequence = request.form['sequence']
#         encode_input = onehot_encoder([input_sequence])
#         model = swin1d_block(4)
#         model.eval()
#         output = model(encode_input)
#         output_array = output.detach().numpy().squeeze()

#         plt.subplot(1, 3, 1)
#         plt.imshow(output_array, cmap='gray')
#         plt.title("Genomic Model Output")

#         white_part = extract_white_part(output_array)
#         plt.subplot(1, 3, 2)
#         plt.imshow(output_array, cmap='gray')
#         plt.title("Original Image")

#         plt.subplot(1, 3, 3)
#         plt.imshow(white_part, cmap='gray')
#         plt.title("White Part")
#         plt.show()

#         visualize_dna_sequence(input_sequence)

#     return render_template('index.html')

# if __name__ == "__main__":
#     app.run(debug=True)
