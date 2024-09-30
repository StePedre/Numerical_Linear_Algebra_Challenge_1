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
• Apply the previous sharpening filter to the original image by performing the matrix vector
multiplication A2v. Export and upload the resulting image.
*/


int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <image_path>" << endl;
        return 1;
    }

    const char* input_image_path = argv[1];

    // Load the image using stb_image
    int width, height, channels;
    // for greyscale images force to load only one channel
    unsigned char* image_data = stbi_load(input_image_path, &width, &height, &channels, 1);
    if (!image_data) {
        cerr << "Error: Could not load image " << input_image_path << endl;
        return 1;
    }
    cout << "Image loaded: " << width << "x" << height << " with " << channels << " channels." << endl;
    
    // Prepare Eigen matrices for rotated picture
    MatrixXd original(height, width);

    // Fill the matrices with image data
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int index = (i * width + j) * channels;  // 1 channel (Greyscale) 3 channels (RGB)
            original(i,j) = static_cast<double>(image_data[index]) / 255;
        }
    }
    // Free memory!!!
    stbi_image_free(image_data);

    // BUILD THE A2 MATRIX OF THE FILTER sh2
    MatrixXd kernel(3, 3);
    kernel << 0.0, -3.0, 0.0,
                -1.0, -9.0, -1.0,
                0.0, -1.0, 0.0;
    SparseMatrix<double, RowMajor> A2(original.size(), original.size());
    for(int i=0; i<original.rows(); i++){
        for(int j=0; j<original.cols(); j++){
            int row_index = (i*original.cols())+j;
            for(int v=0; v<kernel.rows(); v++){
                for(int w=0; w<kernel.cols(); w++){
                    int col_index = ((v - kernel.rows()/2) * original.cols()) + (w - kernel.cols()/2) + row_index;
                    int numberOfRow = row_index/original.cols();
                    int left_extreme = ((v - kernel.rows()/2)*original.cols()) + (original.cols()*numberOfRow);
                    int right_extreme = ((v - kernel.rows()/2)*original.cols()) + (original.cols()*numberOfRow)  + original.cols();                    
                    if(col_index >= max(left_extreme, 0) && col_index < min(right_extreme, (int) A2.cols()) ){
                        A2.insert(row_index, col_index) = kernel(v, w);
                    }
                }
            }
        }
    }
    A2.makeCompressed();

    Matrix<double, Dynamic, Dynamic, RowMajor> image_row_major = original;
    VectorXd image_vector = Map<VectorXd>(image_row_major.data(), image_row_major.size());
    MatrixXd multiplication = A2 * image_vector;
    Matrix<double, Dynamic, Dynamic, RowMajor> multiplication_row_major = multiplication;
    VectorXd multiplicationVector = Map<VectorXd>(multiplication_row_major.data(), multiplication_row_major.size());
    Matrix<double, Dynamic, Dynamic, RowMajor> sharpened_matrix = Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(multiplicationVector.data(), height, width);

    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> sharpened_image(height, width);
    sharpened_image = sharpened_matrix.unaryExpr([](double val) -> unsigned char {
        return static_cast<unsigned char>(val * 255.0);
    });
    
    const string output_image_path = "sharpened_image.png";
    if (stbi_write_png(output_image_path.c_str(), width, height, 1, sharpened_image.data(), width) == 0) {
        cerr << "Error: Could not save rotated image" << endl;
        return 1;
    }

    cout << "Sharpened image saved to " << output_image_path << endl;
    return 0;
}