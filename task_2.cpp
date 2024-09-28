#include <Eigen/Dense>
#include <iostream>
#include <cstdlib>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;
using namespace Eigen;

// Funzione che aggiunge il rumore a un singolo valore di pixel
double addNoise(double val) {
    int noise = rand() % 101 - 50;
    return min(max(val + noise, 0.0), 255.0);
}

// Funzione che converte un valore double in unsigned char
unsigned char toUnsignedChar(double val) {
    return static_cast<unsigned char>(val);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <image_path>" << endl;
        return 1;
    }

    const char* input_image_path = argv[1];

    int width, height, channels;
    unsigned char* image_data = stbi_load(input_image_path, &width, &height, &channels, 1);  // Caricare come scala di grigi

    if (!image_data) {
        cerr << "Error: Could not load image " << input_image_path << endl;
        return 1;
    }

    cout << "Image loaded: " << width << "x" << height << " with " << channels << " channels." << endl;

    MatrixXd image_matrix(height, width);

    // Riempire la matrice con i dati dell'immagine
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int index = i * width + j;
            image_matrix(i, j) = static_cast<double>(image_data[index]);
        }
    }

    // Aggiungere il rumore casuale [-50, 50] a ciascun pixel usando la funzione addNoise
    MatrixXd noisy_image = image_matrix.unaryExpr(&addNoise);

    // Convertire l'immagine con rumore in formato unsigned char usando la funzione toUnsignedChar
    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> noisy_image_char(height, width);
    noisy_image_char = noisy_image.unaryExpr(&toUnsignedChar);

    // Salvare l'immagine risultante
    const string output_image_path = "noisy_image.png";
    if (stbi_write_png(output_image_path.c_str(), width, height, 1, noisy_image_char.data(), width) == 0) {
        cerr << "Error: Could not save noisy image" << endl;
        return 1;
    }

    cout << "Noisy image saved to " << output_image_path << endl;

    // Libera la memoria usata per l'immagine originale
    stbi_image_free(image_data);

    return 0;
}
