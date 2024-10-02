#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace Eigen;
using namespace std;

// Funzione per convertire un valore double in unsigned char
unsigned char toUnsignedChar(double val) {
    return static_cast<unsigned char>(min(max(val * 255.0, 0.0), 255.0));
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <noisy_image_path>" << endl;
        return 1;
    }

    const char* noisy_image_path = argv[1];

    // CARICAMENTO DELLA NOYSE IMMAGE (DALLA TASK 3)
    int width, height, channels;
    unsigned char* noisy_image_data = stbi_load(noisy_image_path, &width, &height, &channels, 1);  // 1 forza scala di grigi
    if (!noisy_image_data) {
        cerr << "Error: Could not load image " << noisy_image_path << endl;
        return 1;
    }

    cout << "Noisy image loaded: " << width << "x" << height << " with " << channels << " channels." << endl;

    // Convertire l'immagine NOYSE in un vettore w
    VectorXd w(width * height);
    for (int i = 0; i < width * height; ++i) {
        w(i) = static_cast<double>(noisy_image_data[i]) / 255.0;
    }

    stbi_image_free(noisy_image_data);  // #FREE MEMORY

    // GENERAZIONE DELLA MATRICE A1 (DALLA TASK 4)
    MatrixXd kernel(3, 3);
    kernel.setConstant(1.0 / 9.0);
    SparseMatrix<double, RowMajor> A1(width * height, width * height);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int row_index = (i * width) + j;
            for (int v = 0; v < kernel.rows(); v++) {
                for (int w = 0; w < kernel.cols(); w++) {
                    int col_index = ((v - kernel.rows() / 2) * width) + (w - kernel.cols() / 2) + row_index;
                    int numberOfRow = row_index / width;
                    int left_extreme = ((v - kernel.rows() / 2) * width) + (width * numberOfRow);
                    int right_extreme = ((v - kernel.rows() / 2) * width) + (width * numberOfRow) + width;
                    if (col_index >= max(left_extreme, 0) && col_index < min(right_extreme, (int)A1.cols())) {
                        A1.insert(row_index, col_index) = kernel(v, w);
                    }
                }
            }
        }
    }
    A1.makeCompressed();

    // APPLICAZIONE DEL FILTRO DI SMOOTHING
    VectorXd smoothed_w = A1 * w;

    // RICONVERSIONE IN MATRICE 2D E SALVATAGGIO
    MatrixXd smoothed_image(height, width);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            smoothed_image(i, j) = smoothed_w(i * width + j);
        }
    }

    // Convertire la matrice smoothed_image in unsigned char
    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> smoothed_image_char(height, width);
    smoothed_image_char = smoothed_image.unaryExpr(&toUnsignedChar);

    // Salvare l'immagine smoothed in formato PNG
    const char* output_image_path = "smoothed_image.png";
    if (stbi_write_png(output_image_path, width, height, 1, smoothed_image_char.data(), width) == 0) {
        cerr << "Error: Could not save smoothed image" << endl;
        return 1;
    }

    cout << "Smoothed image saved to " << output_image_path << endl;

    return 0;
}
