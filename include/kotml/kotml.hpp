#pragma once

// Main KotML components
#include "kotml/tensor.hpp"

// Operations
#include "kotml/ops/basic.hpp"
#include "kotml/ops/linalg.hpp"
#include "kotml/ops/reduction.hpp"

// Neural networks
#include "kotml/nn/module.hpp"
#include "kotml/nn/activations.hpp"
#include "kotml/nn/layers.hpp"
#include "kotml/nn/loss.hpp"

// Optimizers
#include "kotml/optim/optimizer.hpp"
#include "kotml/optim/sgd.hpp"
#include "kotml/optim/adam.hpp"

// Utilities
#include "kotml/utils/helpers.hpp"
#include "kotml/utils/progress_bar.hpp"

namespace kotml {
    // Library version
    constexpr const char* VERSION = "0.1.0";
    
    // Main namespaces
    using namespace kotml::nn;
    using namespace kotml::optim;
    using namespace kotml::utils;
} 