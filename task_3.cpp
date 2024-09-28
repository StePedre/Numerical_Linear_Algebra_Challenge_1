#include <Eigen/Dense>
#include <iostream>
#include <cstdlib>

// from https://github.com/nothings/stb/tree/master
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

//Reshape the original and noisy images as vectors v and w, respectively. Verify that each vector has m n components. Report here the Euclidean norm of v.


using namespace Eigen;

unsigned char* load_image_data(const char* image_path, int &width, int &height, int &channels){
    unsigned char* image_data = stbi_load(image_path, &width, &height, &channels, 1);
    if (!image_data) {
        std::cerr << "Error: Could not load image " << image_path << std::endl;
        return NULL;
    }
    return image_data;
}
using namespace std;

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <image_path> <image_path>"<< std::endl;
        return 1;
    }

    const char* original_image_path = argv[1];
    const char* noisy_image_path = argv[2];

    // Load the image using stb_image
    int width, height, channels;
    unsigned char* original_image_data = load_image_data(original_image_path, width, height, channels);
    std::cout << "Original image loaded: " << width << "x" << height << " with " << channels << " channels." << std::endl;

    unsigned char* noisy_image_data = load_image_data(noisy_image_path, width, height, channels);
    std::cout << "Noisy image loaded: " << width << "x" << height << " with " << channels << " channels." << std::endl;

    VectorXd v(width*height), w(width*height);

    for(int i=0; i<width*height; i++){
        v(i) = static_cast<double>(original_image_data[i]) / 255.0;
        w(i) = static_cast<double>(noisy_image_data[i]) / 255.0;
    }
    stbi_image_free(original_image_data);
    stbi_image_free(noisy_image_data);

    cout << "Vector v size: " << v.size() << endl;
    cout << "Vector w size: " << w.size() << endl;

    cout << "Euclidean norm of v: " << v.norm() << endl;    
}

