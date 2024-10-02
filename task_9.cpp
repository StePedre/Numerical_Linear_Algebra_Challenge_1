#include <Eigen/Dense>
#include <iostream>
#include <cmath>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace Eigen;
using namespace std;

// Funzione per convertire un valore double in unsigned char
unsigned char toUnsignedChar(double val) {
    return static_cast<unsigned char>(min(max(val * 255.0, 0.0), 255.0));
}

int main() {
    // ASPETTO TE ANGELUS
    VectorXd x;

    // SUPPONGO CHE L'IMMAGINE SIA QUADRATA
    int size = x.size();
    int width = static_cast<int>(sqrt(size));
    int height = width;  

    // Verifico che width * height corrisponda esattamente alla dimensione di x
    if (width * height != size) {
        cerr << "Error: The dimensions of the image do not match the size of the vector x." << endl;
        return 1;
    }

    // Riconvertire il vettore x in una matrice 2D
    MatrixXd image_matrix(height, width);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            image_matrix(i, j) = x(i * width + j);
        }
    }

    // Convertire la matrice in unsigned char per salvare l'immagine
    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> image_char(height, width);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            image_char(i, j) = toUnsignedChar(image_matrix(i, j));
        }
    }

    // Salvare l'immagine in formato PNG
    const char* output_image_path = "solution_image.png";
    if (stbi_write_png(output_image_path, width, height, 1, image_char.data(), width) == 0) {
        cerr << "Error: Could not save image" << endl;
        return 1;
    }

    cout << "Image saved to " << output_image_path << endl;

    return 0;
}
