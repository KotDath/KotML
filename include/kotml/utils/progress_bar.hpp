#pragma once

#include <iostream>
#include <string>
#include <iomanip>
#include <sstream>

namespace kotml {
namespace utils {

/**
 * Progress bar utility for training visualization
 * Displays progress in format:
 * Epoch 5 / 12
 * [=======............] 65 / 100 loss: 1.488
 */
class ProgressBar {
private:
    int m_totalEpochs;
    int m_currentEpoch;
    int m_totalSamples;
    int m_currentSample;
    int m_barWidth;
    float m_currentLoss;
    bool m_showSamples;
    
public:
    /**
     * Constructor
     * @param totalEpochs Total number of epochs
     * @param totalSamples Total number of samples per epoch (0 to hide sample progress)
     * @param barWidth Width of the progress bar (default: 20)
     */
    explicit ProgressBar(int totalEpochs, int totalSamples = 0, int barWidth = 20);
    
    /**
     * Update progress for epoch-only training (full batch)
     * @param epoch Current epoch number
     * @param loss Current loss value
     */
    void Update(int epoch, float loss);
    
    /**
     * Update progress for mini-batch training
     * @param epoch Current epoch number
     * @param sample Current sample number within epoch
     * @param loss Current loss value
     */
    void Update(int epoch, int sample, float loss);
    
    /**
     * Finish current epoch (for mini-batch training)
     */
    void FinishEpoch();
    
    /**
     * Finish progress bar and add final newline
     */
    void Finish();
    
private:
    /**
     * Display the current progress bar
     */
    void Display();
};

} // namespace utils
} // namespace kotml 