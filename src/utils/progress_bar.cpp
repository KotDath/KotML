#include "kotml/utils/progress_bar.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>

namespace kotml {
namespace utils {

ProgressBar::ProgressBar(int totalEpochs, int totalSamples, int barWidth)
    : m_totalEpochs(totalEpochs), m_currentEpoch(0), m_totalSamples(totalSamples), 
      m_currentSample(0), m_barWidth(barWidth), m_currentLoss(0.0f), 
      m_showSamples(totalSamples > 0) {
}

void ProgressBar::Update(int epoch, float loss) {
    m_currentEpoch = epoch;
    m_currentLoss = loss;
    m_showSamples = false;
    Display();
}

void ProgressBar::Update(int epoch, int sample, float loss) {
    m_currentEpoch = epoch;
    m_currentSample = sample;
    m_currentLoss = loss;
    m_showSamples = true;
    Display();
}

void ProgressBar::FinishEpoch() {
    if (m_showSamples) {
        m_currentSample = m_totalSamples;
        Display();
        std::cout << std::endl;
    }
}

void ProgressBar::Finish() {
    if (m_showSamples && m_currentSample < m_totalSamples) {
        m_currentSample = m_totalSamples;
        Display();
    }
    std::cout << std::endl;
}

void ProgressBar::Display() {
    // Clear current line and move cursor to beginning
    std::cout << "\r";
    
    if (m_showSamples && m_totalSamples > 0) {
        // Mini-batch training: show epoch and sample progress in one line
        // Format: "Epoch 5 / 12 [=======............] 65 / 100 loss: 1.488"
        
        // Calculate progress for samples within epoch
        float progress = static_cast<float>(m_currentSample) / static_cast<float>(m_totalSamples);
        int filledWidth = static_cast<int>(progress * m_barWidth);
        
        // Display epoch information and progress bar
        std::cout << "Epoch " << m_currentEpoch << " / " << m_totalEpochs << " ";
        
        // Create progress bar
        std::cout << "[";
        for (int i = 0; i < m_barWidth; ++i) {
            if (i < filledWidth) {
                std::cout << "=";
            } else {
                std::cout << ".";
            }
        }
        std::cout << "] ";
        
        // Display sample progress and loss
        std::cout << m_currentSample << " / " << m_totalSamples;
        std::cout << " loss: " << std::fixed << std::setprecision(3) << m_currentLoss;
    } else {
        // Full batch training: show epoch progress in one line
        // Format: "Epoch 5 / 12 [=======............] loss: 1.488"
        
        float progress = static_cast<float>(m_currentEpoch) / static_cast<float>(m_totalEpochs);
        int filledWidth = static_cast<int>(progress * m_barWidth);
        
        // Display epoch information and progress bar
        std::cout << "Epoch " << m_currentEpoch << " / " << m_totalEpochs << " ";
        
        // Create progress bar
        std::cout << "[";
        for (int i = 0; i < m_barWidth; ++i) {
            if (i < filledWidth) {
                std::cout << "=";
            } else {
                std::cout << ".";
            }
        }
        std::cout << "] ";
        
        // Display loss
        std::cout << "loss: " << std::fixed << std::setprecision(3) << m_currentLoss;
    }
    
    std::cout << std::flush;
}

} // namespace utils
} // namespace kotml 