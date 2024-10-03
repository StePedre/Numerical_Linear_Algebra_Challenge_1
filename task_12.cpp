#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <iostream>

using namespace Eigen;
using namespace std;

// Funzione per costruire la matrice A3 (dalla task 11)
SparseMatrix<double> buildA3(const MatrixXd& original) {
    MatrixXd kernel(3, 3);
    kernel << 0.0, -1.0, 0.0,
              -1.0,  4.0, -1.0,
               0.0, -1.0, 0.0;

    SparseMatrix<double, RowMajor> A3(original.size(), original.size());
    for (int i = 0; i < original.rows(); i++) {
        for (int j = 0; j < original.cols(); j++) {
            int row_index = (i * original.cols()) + j;
            for (int v = 0; v < kernel.rows(); v++) {
                for (int w = 0; w < kernel.cols(); w++) {
                    int col_index = ((v - kernel.rows() / 2) * original.cols()) + (w - kernel.cols() / 2) + row_index;
                    int numberOfRow = row_index / original.cols();
                    int left_extreme = ((v - kernel.rows() / 2) * original.cols()) + (original.cols() * numberOfRow);
                    int right_extreme = ((v - kernel.rows() / 2) * original.cols()) + (original.cols() * numberOfRow) + original.cols();
                    if (col_index >= max(left_extreme, 0) && col_index < min(right_extreme, (int) A3.cols())) {
                        A3.insert(row_index, col_index) = kernel(v, w);
                    }
                }
            }
        }
    }
    A3.makeCompressed();
    return A3;
}

// Funzione per caricare il vettore w (dalla task 3)
VectorXd loadW(const char* noisy_image_path, int& width, int& height) {
    int channels;
    unsigned char* noisy_image_data = stbi_load(noisy_image_path, &width, &height, &channels, 1); // Scala di grigi
    if (!noisy_image_data) {
        cerr << "Error: Could not load image " << noisy_image_path << endl;
        exit(1);
    }

    VectorXd w(width * height);
    for (int i = 0; i < width * height; i++) {
        w(i) = static_cast<double>(noisy_image_data[i]) / 255.0;
    }

    stbi_image_free(noisy_image_data); // Free memory
    return w;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <noisy_image_path>" << endl;
        return 1;
    }

    const char* noisy_image_path = argv[1];
    int width, height;

    // Caricare il vettore w dalla noyse immage dalla task 3
    VectorXd w = loadW(noisy_image_path, width, height);

    // Costruire la matrice A3 dalla task 11
    MatrixXd original_image(height, width); 
    SparseMatrix<double> A3 = buildA3(original_image);

    // Costruire la matrice I + A3
    SparseMatrix<double> I(A3.rows(), A3.cols());
    I.setIdentity();  // Matrice identità di dimensioni uguali a A3
    SparseMatrix<double> I_plus_A3 = I + A3;

    ConjugateGradient<SparseMatrix<double>, Lower|Upper> solver;
    solver.setTolerance(1e-10);  // Impostare la tolleranza a 10^-10
    solver.compute(I_plus_A3);

    if (solver.info() != Success) {
        cerr << "Decomposition failed!" << endl;
        return 1;
    }

    // Risolvere il sistema (I + A3)y = w
    VectorXd y = solver.solve(w);

    if (solver.info() != Success) {
        cerr << "Solving failed!" << endl;
        return 1;
    }

    // # iterazioni e residual
    cout << "Number of iterations: " << solver.iterations() << endl;
    cout << "Final residual: " << solver.error() << endl;

    return 0;
}
