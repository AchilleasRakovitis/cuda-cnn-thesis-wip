#pragma once
#include <string>
#include <vector>
#include <cstdint>

// function declarations 
void load_cifar10_batch(const std::string& filename, std::vector<float>& pixels,
                        std::vector<uint8_t>& labels);

void load_cifar10_train(const std::string& folder, std::vector<float>& total_pixels,
                        std::vector<uint8_t>& total_labels);

void load_cifar10_test(const std::string& filename, std::vector<float>& pixels,
                        std::vector<uint8_t>& labels);