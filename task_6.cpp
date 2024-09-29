#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <cstdlib>

// from https://github.com/nothings/stb/tree/master
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

//Write the convolution operation corresponding to the sharpening kernel Hsh2 as a matrix vector multiplication by a matrix A2 having size mn×mn. Report the number of non-zero entries in A2. Is A2 symmetric?


using namespace Eigen;
using namespace std;

VectorXd perform_convolution(MatrixXd &image_data, MatrixXd &kernel){
    MatrixXd A_2(image_data.size(), image_data.size());
    for(int i=0; i<kernel.rows(); i++){
        for(int j=0; j<kernel.cols(); j++){
            int diagonal_num = ((i - (kernel.rows()/2)) * image_data.cols()) + (j - (kernel.cols()/2));
            cout << diagonal_num << " " <<A_2.diagonal(diagonal_num).size() << endl;
            A_2.diagonal(diagonal_num) = VectorXd::Constant(A_2.diagonal(diagonal_num).size(), kernel(i,j));
        }
    }

    SparseMatrix<double, RowMajor> A_2_sparse = A_2.sparseView();

    cout << "Zero entries in A_2: " << A_2_sparse.size() - A_2_sparse.nonZeros() << endl;

    MatrixXd multiplication = A_2_sparse * Map<VectorXd>(image_data.data(), image_data.size()); 
    return Map<VectorXd>(multiplication.data(), image_data.size());
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return 1;
    }

    const char* input_image_path = argv[1];

    // Load the image using stb_image
    int width, height, channels;
    // for greyscale images force to load only one channel
    unsigned char* image_data = stbi_load(input_image_path, &width, &height, &channels, 1);
    if (!image_data) {
        std::cerr << "Error: Could not load image " << input_image_path << std::endl;
        return 1;
    }

    std::cout << "Image loaded: " << width << "x" << height << " with " << channels << " channels." << std::endl;
    
    // Prepare Eigen matrices for rotated picture
    MatrixXd original(height, width);

    // Fill the matrices with image data
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int index = (i * width + j) * channels;  // 1 channel (Greyscale) 3 channels (RGB)
            original(i,j) = static_cast<double>(image_data[index]) / 255.0;
        }
    }
    // Free memory!!!
    stbi_image_free(image_data);

    MatrixXd kernel(3, 3);
    kernel << 0.0, 1.0, 0.0,
                1.0, 1.0, 1.0,
                0.0, 1.0, 0.0;
    kernel *= (1.0/5.0);
    cout << kernel << endl;
    VectorXd sharpened = perform_convolution(original, kernel);
    MatrixXd reshaped = Map<MatrixXd>(sharpened.data(), height, width);

    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> sharpened_image(height, width);
    // Use Eigen's unaryExpr to map the grayscale values (0.0 to 1.0) to 0 to 255
    sharpened_image = reshaped.unaryExpr([](double val) -> unsigned char {
        return static_cast<unsigned char>(val * 255.0);
    });
    
    // Save the grayscale image using stb_image_write
    const std::string output_image_path = "output_sharpened.png";
    if (stbi_write_png(output_image_path.c_str(), width, height, 1, sharpened_image.data(), width) == 0) {
        std::cerr << "Error: Could not save rotated image" << std::endl;

        return 1;
    }

    std::cout << "Sharpened image saved to " << output_image_path << std::endl;

    return 0;  
}

