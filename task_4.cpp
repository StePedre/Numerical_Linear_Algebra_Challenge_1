#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <cstdlib>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace Eigen;
using namespace std;

/*
• Write the convolution operation corresponding to the smoothing kernel Hav2 as a matrix
vector multiplication between a matrix A1 having size mn x mn and the image vector.
Report the number of non-zero entries in A1.
*/

void printSparseMatrix(const SparseMatrix<double>& mat) {
    // Loop over the sparse matrix and print non-zero entries
    for (int k = 0; k < mat.outerSize(); ++k) {
        for (SparseMatrix<double>::InnerIterator it(mat, k); it; ++it) {
            cout << "Row: " << it.row() << ", Column: " << it.col() << ", Value: " << it.value() << endl;
        }
    }
}

int main(int argc, char* argv[]) {
    // LOAD IMAGE FILE
    const char* imagePath = argv[1];
    int width, height, channels;
    unsigned char* imageData = stbi_load(imagePath, &width, &height, &channels, 1);
    if (!imageData) {
        cerr << "Error: could not load the image " << imagePath << "." << endl;
        return 1;
    }

    // SAVES THE IMAGE DATA IN A MATRIX
    MatrixXd imageMatrix(height, width);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int index = i * width + j;
            imageMatrix(i, j) = imageData[index] / 255.0;
        }
    }
    stbi_image_free(imageData);
    cout << "Image loaded correctly. The size of the matrix is " << imageMatrix.rows() << "x" << imageMatrix.cols() << endl;

    // BUILD THE A1 MATRIX OF THE FILTER av2 - TO FIX (TOO SLOW)
    double kernelValue = 1.0 / 9.0;
    SparseMatrix<double> A1(width * height, width * height);
    for (int i = 0; i < width * height; i++) {
        A1.insert(i, i) = kernelValue;
        // Left neighbor
        if (i > 0) A1.insert(i, i - 1) = kernelValue; 
        // Right neighbor
        if (i < width - 1) A1.insert(i, i + 1) = kernelValue; 
        // Top neighbor
        if (i > width) A1.insert(i, i - width) = kernelValue; 
        // Bottom neighbor
        if (i < height - 1) A1.insert(i, i + width) = kernelValue; 
        // Top-left neighbor
        if (i > width) A1.insert(i, i - width - 1) = kernelValue; 
        // Top-right neighbor
        if (i > width && i < width - 1) A1.insert(i, i - width + 1) = kernelValue; 
        // Bottom-left neighbor
        if (i < height - 1 && i > 0) A1.insert(i, i + width - 1) = kernelValue; 
        // Bottom-right neighbor
        if (i < height - 1 && i < width - 1) A1.insert(i, i + width + 1) = kernelValue;
    }
    A1.makeCompressed();

    // DISPLAY THE NON ZERO ENTRIES IN THE A1 MATRIX
    cout << "The number of non-zero entries of A1 matrix is " << A1.nonZeros() << "." << endl;
    //printSparseMatrix(A1);
    return 0;
}