#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <cmath>  // For std::fabs

// from https://github.com/nothings/stb/tree/master
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

//Write the convolution operation corresponding to the sharpening kernel Hsh2 as a matrix vector multiplication by a matrix A2 having size mn×mn. Report the number of non-zero entries in A2. Is A2 symmetric?


using namespace Eigen;
using namespace std;

VectorXd perform_convolution(const MatrixXd &image_data, const MatrixXd &kernel) {
    int rows = image_data.rows();
    int cols = image_data.cols();
    int kernel_rows = kernel.rows();
    int kernel_cols = kernel.cols();

    // Output matrix
    MatrixXd output(rows, cols);
    output.setZero(); // Initialize output with zeros

    int kernel_center_row = kernel_rows / 2;
    int kernel_center_col = kernel_cols / 2;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double sum = 0.0; // Accumulate the convolution result

            // Apply the kernel
            for (int v = 0; v < kernel_rows; v++) {
                for (int w = 0; w < kernel_cols; w++) {
                    int image_row = i + (v - kernel_center_row);
                    int image_col = j + (w - kernel_center_col);

                    // Ensure the pixel is within the image bounds
                    if (image_row >= 0 && image_row < rows && image_col >= 0 && image_col < cols) {
                        sum += image_data(image_row, image_col) * kernel(v, w);
                    }
                }
            }
            sum = std::min(sum, 1.0);
            sum = std::max(sum, 0.0);
            output(i, j) = sum; // Store the result in the output matrix
        }
    }

    // Flatten the output to a VectorXd
    return Map<VectorXd>(output.data(), output.size());
}

VectorXd perform_H_sh_2(const MatrixXd &image_data){
    MatrixXd kernel(3, 3);
    kernel << 0.0, -1.0, 0.0,
                -1.0, 5.0, -1.0,
                0.0, -1.0, 0.0;
    return perform_convolution(image_data, kernel);
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
            original(i,j) = static_cast<double>(image_data[index]) / 255;
        }
    }
    // Free memory!!!
    stbi_image_free(image_data);

    VectorXd sharpened = perform_H_sh_2(original);
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
