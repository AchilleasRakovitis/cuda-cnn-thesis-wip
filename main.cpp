#include "data_loader.h"
#include <iostream>

int main() {
    // Test training data
    std::vector<float> train_pixels;
    std::vector<uint8_t> train_labels;
    
    load_cifar10_train("cifar-10-batches-bin", train_pixels, train_labels);
    
    std::cout << "=== Training Data ===" << std::endl;
    std::cout << "Images: " << train_labels.size() << std::endl;
    std::cout << "Pixel values: " << train_pixels.size() << std::endl;
    
    // Test test data
    std::vector<float> test_pixels;
    std::vector<uint8_t> test_labels;
    
    load_cifar10_test("cifar-10-batches-bin", test_pixels, test_labels);
    
    std::cout << "=== Test Data ===" << std::endl;
    std::cout << "Images: " << test_labels.size() << std::endl;
    std::cout << "Pixel values: " << test_pixels.size() << std::endl;
    
    // Verify some labels
    const std::string class_names[] = {
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    };
    
    std::cout << "=== First 10 training labels ===" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << "Image " << i << ": " << class_names[train_labels[i]] << std::endl;
    }
    
    return 0;
}   


