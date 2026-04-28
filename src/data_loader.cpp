#include "data_loader.h"
#include <fstream>
#include <iostream>

void load_cifar10_batch(const std::string& filename, std::vector<float>& pixels,
                        std::vector<uint8_t>& labels){
    
    //declare constants 
    const int BATCH_SIZE = 10000;
    const int LABEL_SIZE = 10000;
    const int PIXEL_SIZE =  10000 * 32 * 32 * 3;
    constexpr int PIXEL_SIZE_ONE_IMAGE = 3072;
    
    //open binary file
    std::ifstream file(filename, std::ios::binary);
    
    //file opening error checking
    if(!file.is_open()){
        std::cerr << "Could not open file" << std::endl;
        return;
    }

    std::cout << "File opened successfully" << std::endl;

    //resize the vector to the exact length needed
    pixels.resize(PIXEL_SIZE);
    labels.resize(LABEL_SIZE);

    //temporary buffer for the pixel array of each image
    uint8_t pixel_buffer[PIXEL_SIZE_ONE_IMAGE];

    //track how many images have been successfully loaded
    int loaded = 0;

    //loop all the images
    for(int i = 0; i < BATCH_SIZE; i++){
        uint8_t label;

        //read the first byte that is a label
        if(!file.read(reinterpret_cast<char*>(&label), 1)){
            std::cerr << "Error loading label at index" << i << std::endl;
            break;
        }
        //read the next 3072 pixel buffer array 
        if(!file.read(reinterpret_cast<char*>(pixel_buffer), PIXEL_SIZE_ONE_IMAGE)){
            std::cerr << "Error loading image at index" << i << std::endl;
            break;
        };

        //assign labels vector each label byte
        labels[i] = label;

        // index for every start of the new pixel's image
        int base = i * PIXEL_SIZE_ONE_IMAGE;

        //assign and normalize to range 0.0-1.0
        for(int j = 0; j < PIXEL_SIZE_ONE_IMAGE; j++){
            pixels[base+ j] = static_cast<float>(pixel_buffer[j]) / 255.0f;
        }

        //increment the amount of images successfully loaded
        loaded++;
    }

    //resize based on the amount of sucessfully loaded images
    pixels.resize(loaded * PIXEL_SIZE_ONE_IMAGE);
    labels.resize(loaded);

    //close the file
    file.close();

}

void load_cifar10_train(const std::string& folder, std::vector<float>& total_pixels, 
                        std::vector<uint8_t>& total_labels){
    //number of batch files
    const int iters = 5;
    
    ////temporary vectors that get filled by the load_cifar10_batch function (passed by reference)
    std::vector<float> batch_pixels;
    std::vector<uint8_t> batch_labels;

    //loop all the batch files in the folder
    for(int i = 0; i < iters; i++){
        // filename string creation
        std::string filename = folder + "/data_batch_" + std::to_string(i+1) + ".bin";
        //function call
        load_cifar10_batch(filename, batch_pixels, batch_labels);

        //append to the large vector the temporary vectors returned
        total_pixels.insert(total_pixels.end(), batch_pixels.begin(), batch_pixels.end());
        total_labels.insert(total_labels.end(), batch_labels.begin(), batch_labels.end());
    }
}

// call the load_cifar10_batch but for the one testing file same logic as before
void load_cifar10_test(const std::string& folder, std::vector<float>& pixels,
                        std::vector<uint8_t>& labels){
    std::string filename = folder + "/test_batch.bin";

    load_cifar10_batch(filename, pixels, labels);
}