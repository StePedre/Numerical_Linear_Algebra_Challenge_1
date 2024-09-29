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

    // BUILD THE A1 MATRIX OF THE FILTER av2
    MatrixXd kernel(3, 3);
    kernel.setConstant(1.0 / 9.0);
    SparseMatrix<double, RowMajor> A1(imageMatrix.size(), imageMatrix.size());
    for(int i=0; i<imageMatrix.rows(); i++){
        for(int j=0; j<imageMatrix.cols(); j++){
            int row_index = (i*imageMatrix.cols())+j;
            for(int v=0; v<kernel.rows(); v++){
                for(int w=0; w<kernel.cols(); w++){
                    int col_index = ((v - kernel.rows()/2) * imageMatrix.cols()) + (w - kernel.cols()/2) + row_index;
                    int numberOfRow = row_index/imageMatrix.cols();
                    int left_extreme = ((v - kernel.rows()/2)*imageMatrix.cols()) + (imageMatrix.cols()*numberOfRow);
                    int right_extreme = ((v - kernel.rows()/2)*imageMatrix.cols()) + (imageMatrix.cols()*numberOfRow)  + imageMatrix.cols();                    
                    if(col_index >= max(left_extreme, 0) && col_index < min(right_extreme, (int) A1.cols()) ){
                        A1.insert(row_index, col_index) = kernel(v, w);
                    }
                }
            }
        }
    }
    A1.makeCompressed();

    // DISPLAY THE NON ZERO ENTRIES IN THE A1 MATRIX
    cout << "The number of non-zero entries of A1 matrix is " << A1.nonZeros() << "." << endl;
    //printSparseMatrix(A1);
    return 0;
}