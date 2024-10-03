#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <cstdlib>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <unsupported/Eigen/SparseExtra>

using namespace Eigen;
using namespace std;

SparseMatrix<double, RowMajor> generate_convoultion_matrix(const MatrixXd &kernel, int height, int width);

int main(int argc, char *argv[])
{
    /*
                                        TASK 1
        Load the image as an Eigen matrix with size m x n. Each entry in the matrix corresponds
        to a pixel on the screen and takes a value somewhere between 0 (black) and 255 (white).
        Report the size of the matrix.
    */
    cout << "********* TASK 1 *********" << endl;
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return 1;
    }
    const char *image_path = argv[1];
    int width, height, channels;
    unsigned char *image_data = stbi_load(image_path, &width, &height, &channels, 1);
    if (!image_data)
    {
        cerr << "Error: could not load the image " << image_path << "." << endl;
        return 1;
    }

    MatrixXd image_matrix(height, width), noisy_matrix(height, width);
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            int index = (i * width + j) * channels;
            image_matrix(i, j) = image_data[index] / 255.0;
            noisy_matrix(i, j) = min(max(image_data[index] + (rand() % 101 - 50), 0), 255) / 255.0;
        }
    }
    stbi_image_free(image_data);
    cout << "Image loaded correctly. The size of the matrix is " << image_matrix.rows() << "x" << image_matrix.cols() << endl;

    /*
                                        TASK 2
        Introduce a noise signal into the loaded image by adding random fluctuations of color
        ranging between [−50, 50] to each pixel. Export the resulting image in .png and upload it.
    */
    cout << "********* TASK 2 *********" << endl;
    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> noisy_image_char(height, width);
    noisy_image_char = noisy_matrix.unaryExpr([](double val) -> unsigned char
                                              { return static_cast<unsigned char>(val * 255.0); });

    const string output_noisy_path = "output/noisy_image.png";
    if (stbi_write_png(output_noisy_path.c_str(), width, height, 1, noisy_image_char.data(), width) == 0)
    {
        cerr << "Error: Could not save noisy image" << endl;
        return 1;
    }

    cout << "Noisy image saved to " << output_noisy_path << endl;

    /*
                                        TASK 3
        Reshape the original and noisy images as vectors v and w, respectively.
        Verify that each vector has m n components. Report here the Euclidean norm of v.
    */
    cout << "********* TASK 3 *********" << endl;
    VectorXd v(width * height), w(width * height);

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            v(i * width + j) = image_matrix(i, j);
            w(i * width + j) = noisy_matrix(i, j);
        }
    }
    cout << "Number of components of v: " << v.size() << endl;
    cout << "Number of components of w: " << w.size() << endl;

    cout << "Euclidean norm of v: " << v.norm() << endl;

    /*
                                        TASK 4
        Write the convolution operation corresponding to the smoothing kernel
        Hav2 as a matrix vector multiplication between a matrix A1 having size
        mn × mn and the image vector. Report the number of non-zero entries in A1.
    */
    cout << "********* TASK 4 *********" << endl;
    MatrixXd h_av_2(3, 3);
    h_av_2 << 0.0, 1.0, 0.0,
        1.0, 4.0, 1.0,
        0.0, 1.0, 0.0;
    h_av_2 *= (1.0 / 8.0);
    SparseMatrix<double, RowMajor> a_1 = generate_convoultion_matrix(h_av_2, height, width);

    cout << "Zero entries in A_1: " << a_1.size() - a_1.nonZeros() << endl;

    /*
                                        TASK 5
        Apply the previous smoothing filter to the noisy image by performing
        the matrix vector multiplication A1w. Export and upload the resulting image.
    */
    cout << "********* TASK 5 *********" << endl;
    VectorXd a_1_w_result = a_1 * w;

    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> smoothed_image_char(height, width);
    smoothed_image_char = a_1_w_result.unaryExpr([](double val) -> unsigned char
                                                 { return static_cast<unsigned char>(val * 255.0); });

    const string output_smooth_path = "output/smoothed_image.png";
    if (stbi_write_png(output_smooth_path.c_str(), width, height, 1, smoothed_image_char.data(), width) == 0)
    {
        cerr << "Error: Could not save noisy image" << endl;
        return 1;
    }

    cout << "Smoothed image saved to " << output_smooth_path << endl;

    /*
                                        TASK 6
        Write the convolution operation corresponding to the sharpening kernel
        Hsh2 as a matrix vector multiplication by a matrix A2 having size mn×mn.
        Report the number of non-zero entries in A2. Is A2 symmetric?
    */
    cout << "********* TASK 6 *********" << endl;
    MatrixXd h_sh_2(3, 3);
    h_sh_2 << 0.0, -1.0, 0.0,
        -1.0, 5.0, -1.0,
        0.0, -1.0, 0.0;
    SparseMatrix<double, RowMajor> a_2 = generate_convoultion_matrix(h_sh_2, height, width);

    cout << "Zero entries in A_2: " << a_2.size() - a_2.nonZeros() << endl;
    cout << "Is A_2 symmetric? : " << (a_2.isApprox(a_2.transpose()) == 1 ? "true" : "false") << endl;

    /*
                                        TASK 7
        Apply the previous sharpening filter to the original image by performing
        the matrix vector multiplication A2v. Export and upload the resulting image.
    */
    cout << "********* TASK 7 *********" << endl;
    VectorXd a_2_v_result = a_2 * v;

    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> sharpened_image_char(height, width);
    sharpened_image_char = a_2_v_result.unaryExpr([](double val) -> unsigned char
                                                  { return static_cast<unsigned char>(max(0.0, min(val, 1.0)) * 255.0); });

    const string output_sharp_path = "output/sharpened_image.png";
    if (stbi_write_png(output_sharp_path.c_str(), width, height, 1, sharpened_image_char.data(), width) == 0)
    {
        cerr << "Error: Could not save noisy image" << endl;
        return 1;
    }

    cout << "Sharpened image saved to " << output_sharp_path << endl;

    /*
                                        TASK 8
        Export the Eigen matrix A2 and vector w in the .mtx format. Using a suitable i
        terative solver and preconditioner technique available in the LIS library compute 
        the approximate solution to the linear system A2x = w prescribing a tolerance of 10−9. 
        eport here the iteration count and the final residual.

    */
    cout << "********* TASK 8 *********" << endl;

    const string output_a_2_market_path = "output/a_2.mtx";
    remove(output_a_2_market_path.c_str());
    FILE* out = fopen(output_a_2_market_path.c_str(), "w");
    if (!out) {
        std::cerr << "Error opening file: " <<  output_a_2_market_path << std::endl;
        return -1;
    }
    fprintf(out, "%%%%MatrixMarket matrix coordinate real general\n");
    fprintf(out, "%d %d %d\n", a_2.rows(), a_2.cols(), a_2.nonZeros());
    for (int k = 0; k < a_2.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double, RowMajor>::InnerIterator it(a_2, k); it; ++it) {
    
            fprintf(out, "%d %d  %.20e\n", it.row() + 1, it.col() + 1, it.value());
        }
    }
    fclose(out);
    cout << "A_2 matrix saved to " << output_a_2_market_path << endl;

    const string output_w_market_path = "output/w.mtx";
    remove(output_w_market_path.c_str());
    out = fopen(output_w_market_path.c_str(),"w");
    fprintf(out,"%%%%MatrixMarket vector coordinate real general\n");
    fprintf(out,"%d\n", w.size());
    for (int i=0; i<w.size(); i++) {
       fprintf(out,"%d %.20e\n", i ,w(i));
    }
    fclose(out);
    cout << "W vector saved to " << output_w_market_path << endl;

    const string test_1_path ="./resource/lis-2.0.34/test/test1";
    const string result_path = "output/x.mtx";
    const string history_path = "output/history";
    const string tollerance = "-tol 1.0e-9";
    const string command = test_1_path + " " + output_a_2_market_path + " " + output_w_market_path + " " + result_path + " " + history_path + " " + tollerance;
    system(command.c_str());

    /*
                                        TASK 9
        Import the previous approximate solution vector x in Eigen and 
        then convert it into a .png image. Upload the resulting file here.
    */
    cout << "********* TASK 9 *********" << endl;
    std::ifstream file(result_path);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << result_path << std::endl;
        return -1;
    }
    std::string line;
    std::getline(file, line); // Read the first line (header)

    // Check if the header matches the expected format
    if (line != "%%MatrixMarket vector coordinate real general") {
        std::cerr << "Invalid Matrix Market format!" << std::endl;
        return -1;
    }
    // Read dimensions
    std::getline(file, line);
    std::istringstream iss(line);
    int nrows; // Number of rows and number of non-zero entries
    iss >> nrows;

    VectorXd x(nrows);
    for (int i = 0; i < nrows; ++i) {
        std::getline(file, line);
        std::istringstream entryStream(line);
        int index;
        double value;
        entryStream >> index >> value; // Read index and value
        x(index-1) = value; // Store only the value
    }

    // Close the file
    file.close();

    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> x_image_char(height, width);
    x_image_char =x.unaryExpr([](double val) -> unsigned char
                                                  { return static_cast<unsigned char>(max(0.0, min(val, 1.0)) * 255.0); });

    const string output_x_path = "output/x_image.png";
    if (stbi_write_png(output_x_path.c_str(), width, height, 1, x_image_char.data(), width) == 0)
    {
        cerr << "Error: Could not save noisy image" << endl;
        return 1;
    }

    cout << "Sharpened image saved to " << output_x_path << endl;
    
    
   
    return 0;
}

SparseMatrix<double, RowMajor> generate_convoultion_matrix(const MatrixXd &kernel, int height, int width)
{
    int size = height * width;
    SparseMatrix<double, RowMajor> convoultion_matrix(size, size);
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            int row_index = (i * width) + j;
            for (int v = 0; v < kernel.rows(); v++)
            {
                for (int w = 0; w < kernel.cols(); w++)
                {
                    int col_index = ((v - kernel.rows() / 2) * width) + (w - kernel.cols() / 2) + row_index;
                    int numberOfRow = row_index / width;
                    int left_extreme = ((v - kernel.rows() / 2) * width) + (width * numberOfRow);
                    int right_extreme = ((v - kernel.rows() / 2) * width) + (width * numberOfRow) + width;
                    if (col_index >= max(left_extreme, 0) && col_index < min(right_extreme, (int)convoultion_matrix.cols()))
                    {
                        if(kernel(v, w) != 0.0){
                            convoultion_matrix.insert(row_index, col_index) = kernel(v, w);
                        }
                    }
                }
            }
        }
    }
    convoultion_matrix.makeCompressed();

    return convoultion_matrix;
}