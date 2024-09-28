#include <Eigen/Dense>
#include <iostream>
#include <cstdlib>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace Eigen;
using namespace std;

/*
• Load the image as an Eigen matrix with size m x n. Each entry in the matrix corresponds
to a pixel on the screen and takes a value somewhere between 0 (black) and 255 (white).
Report the size of the matrix.
*/

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
    return 0;
}