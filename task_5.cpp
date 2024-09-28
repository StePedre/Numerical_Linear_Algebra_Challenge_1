#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace Eigen;
using namespace std;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <noisy_image_path>" << endl;
        return 1;
    }

    const char* noisy_image_path = argv[1];

    // CARICAMENTO NOISY IMMAGE (DALLA TASK 3)
    int width, height, channels;
    unsigned char* noisy_image_data = stbi_load(noisy_image_path, &width, &height, &channels, 1);  // 1 forza scala di grigi
    if (!noisy_image_data) {
        cerr << "Error: Could not load image " << noisy_image_path << endl;
        return 1;
    }

    cout << "Noisy image loaded: " << width << "x" << height << " with " << channels << " channels." << endl;

    // Convertire la noyse immage in un vettore w
    VectorXd w(width * height);
    for (int i = 0; i < width * height; ++i) {
        w(i) = static_cast<double>(noisy_image_data[i]) / 255.0;
    }

    stbi_image_free(noisy_image_data);  // FREE MEMORY

    // ===== INSERIMENTO DELLA MATRICE A1 DALLA TASK 4 =====
    SparseMatrix<double> A1(width * height, width * height);

    /* 
    ASPETTO TE STE :;
    */

    // APPLICAZIONE DEL FILTRO DI SMOOTHING
    VectorXd smoothed_w = A1 * w;

    // RICONVERSIONE IN MATRICE 2D E SALVATAGGIO
    // Riconvertire il vettore smoothed_w in una matrice 2D
    MatrixXd smoothed_image(height, width);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            smoothed_image(i, j) = smoothed_w(i * width + j);
        }
    }

    // Convertire la matrice smoothed_image in unsigned char per salvare l'immagine
    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> smoothed_image_char(height, width);
    smoothed_image_char = smoothed_image.unaryExpr([](double val) -> unsigned char {
        return static_cast<unsigned char>(min(max(val * 255.0, 0.0), 255.0));  // Valori tra 0 e 255
    });

    // Salvare l'immagine smoothed in formato PNG
    const char* output_image_path = "smoothed_image.png";
    if (stbi_write_png(output_image_path, width, height, 1, smoothed_image_char.data(), width) == 0) {
        cerr << "Error: Could not save smoothed image" << endl;
        return 1;
    }

    cout << "Smoothed image saved to " << output_image_path << endl;

    return 0;
}
