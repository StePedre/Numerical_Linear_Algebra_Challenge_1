#include <iostream>
#include <Eigen/Sparse>

using namespace std;
using namespace Eigen;
typedef Eigen::Triplet<double> T;

//Triplet List is a list of triplets (i,j,value) that represent the non-zero elements of a sparse matrix

int main(int argc, char** argv)
{
       SparseMatrix<double, RowMajor> mat(10,10);       // define matrix
    // for (int i=0; i<10; i++) {
    //     mat.coeffRef(i, i) = 1.0;
    // }

    // Alternative way to fill the matrix
    std::vector<T> tripletList;
    tripletList.reserve(10);
    for(int i=0; i<10; i++) {
      tripletList.push_back(T(i, i, 1.0));
    }
    mat.setFromTriplets(tripletList.begin(), tripletList.end());

    VectorXd b = VectorXd::Constant(mat.rows(), 1); // define right-hand side

    // Solving 
    SimplicialLDLT<Eigen::SparseMatrix<double> > solver(mat);   // factorization 
    solver.compute(mat);
    if(solver.info()!=Success) {                                // sanity check 
        cout << "cannot factorize the matrix" << endl;          
        return 0;
    }
    
    VectorXd x = solver.solve(b);                   // solving
    cout << x << endl;                              // display solution
    return 0;    
}