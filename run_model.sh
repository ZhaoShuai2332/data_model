#!/bin/bash
# ==============================================================================
# Model Runner Script for Machine Learning Classification Project
# ==============================================================================
# 
# Description:
#   This script serves as the unified entry point for running various machine
#   learning models in the data classification project. It supports multiple
#   model architectures including CNN, MLP, Logistic Regression, SVM, and
#   Transformer-based approaches.
#
# Author: Data Modeling Team
# Date: December 2025
# Version: 1.0
#
# Usage:
#   ./run_model.sh <model_name>      Run a specific model
#   ./run_model.sh all               Run all models sequentially
#   ./run_model.sh help              Display help information
#
# Available Models:
#   cnn          - Convolutional Neural Network (ResNet1D architecture)
#   mlp          - Multi-Layer Perceptron
#   logistic     - Logistic Regression classifier
#   svm          - Support Vector Machine
#   transformer  - Transformer-based model
#
# Examples:
#   ./run_model.sh cnn               # Train and evaluate CNN model
#   ./run_model.sh logistic          # Train and evaluate Logistic Regression
#   ./run_model.sh all               # Run all models for comparison
#
# ==============================================================================

# ==============================================================================
# Environment Configuration
# ==============================================================================

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# ==============================================================================
# Color Definitions for Terminal Output
# ==============================================================================

GREEN='\033[0;32m'      # Success messages
RED='\033[0;31m'        # Error messages
YELLOW='\033[1;33m'     # Warning/Info messages
BLUE='\033[0;34m'       # Section headers
CYAN='\033[0;36m'       # Highlights
NC='\033[0m'            # No Color (reset)

# ==============================================================================
# Utility Functions
# ==============================================================================

# ------------------------------------------------------------------------------
# Function: print_header
# Description: Print a formatted section header
# Arguments:
#   $1 - Header text to display
# ------------------------------------------------------------------------------
print_header() {
    local text="$1"
    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}  $text${NC}"
    echo -e "${BLUE}============================================================${NC}"
    echo ""
}

# ------------------------------------------------------------------------------
# Function: print_usage
# Description: Display usage information and available options
# ------------------------------------------------------------------------------
print_usage() {
    echo ""
    echo "================================================================================"
    echo "  Machine Learning Model Runner - Usage Guide"
    echo "================================================================================"
    echo ""
    echo "SYNOPSIS:"
    echo "    ./run_model.sh <model_name>"
    echo ""
    echo "AVAILABLE MODELS:"
    echo "    cnn          - Convolutional Neural Network (ResNet1D architecture)"
    echo "    mlp          - Multi-Layer Perceptron"
    echo "    logistic     - Logistic Regression classifier"
    echo "    svm          - Support Vector Machine"
    echo "    transformer  - Transformer-based model"
    echo "    all          - Run all models sequentially"
    echo ""
    echo "OPTIONS:"
    echo "    help, -h, --help    Display this help message"
    echo ""
    echo "EXAMPLES:"
    echo "    ./run_model.sh cnn           # Train CNN model"
    echo "    ./run_model.sh logistic      # Train Logistic Regression model"
    echo "    ./run_model.sh all           # Run all models for comparison"
    echo ""
    echo "OUTPUT:"
    echo "    Results will be saved in respective model directories."
    echo "    Comparison metrics will be displayed after execution."
    echo ""
    echo "================================================================================"
}

# ==============================================================================
# Model Execution Functions
# ==============================================================================

# ------------------------------------------------------------------------------
# Function: run_model
# Description: Execute a specific machine learning model
# Arguments:
#   $1 - Model identifier (cnn, mlp, logistic, svm, transformer)
# Returns:
#   0 on success, 1 on failure
# ------------------------------------------------------------------------------
run_model() {
    local model=$1
    local script=""
    local name=""
    
    # Map model identifier to script path and display name
    case $model in
        cnn)
            script="cnn_model/execute.py"
            name="CNN (ResNet1D)"
            ;;
        mlp)
            script="mlp_model/execute.py"
            name="Multi-Layer Perceptron (MLP)"
            ;;
        logistic)
            script="logistic_model/execute.py"
            name="Logistic Regression"
            ;;
        svm)
            script="svm_model/execute.py"
            name="Support Vector Machine (SVM)"
            ;;
        transformer)
            script="transformer_model/execute.py"
            name="Transformer"
            ;;
        *)
            echo -e "${RED}[ERROR] Unknown model: '$model'${NC}"
            echo ""
            echo "Run './run_model.sh help' for a list of available models."
            return 1
            ;;
    esac
    
    # Verify script exists before execution
    if [ ! -f "$script" ]; then
        echo -e "${RED}[ERROR] Script not found: $script${NC}"
        return 1
    fi
    
    # Display execution header
    print_header "Executing $name Model"
    
    echo -e "${CYAN}[INFO] Script: $script${NC}"
    echo -e "${CYAN}[INFO] Start Time: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
    echo ""
    
    # Execute the Python script
    python "$script"
    local exit_code=$?
    
    # Report execution result
    echo ""
    echo -e "${CYAN}[INFO] End Time: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
    
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}[SUCCESS] $name model execution completed successfully!${NC}"
        return 0
    else
        echo -e "${RED}[FAILED] $name model execution failed with exit code: $exit_code${NC}"
        return 1
    fi
}

# ------------------------------------------------------------------------------
# Function: run_all_models
# Description: Execute all available models sequentially and generate summary
# Returns:
#   0 if all models succeed, 1 if any model fails
# ------------------------------------------------------------------------------
run_all_models() {
    print_header "Running All Models - Comparative Evaluation"
    
    echo -e "${CYAN}[INFO] This will execute all available models sequentially.${NC}"
    echo -e "${CYAN}[INFO] Total models to run: 5${NC}"
    echo ""
    
    # Define model list
    local models=("cnn" "mlp" "logistic" "svm" "transformer")
    local model_names=("CNN (ResNet1D)" "MLP" "Logistic Regression" "SVM" "Transformer")
    declare -A results
    local failed_count=0
    
    # Execute each model
    local index=0
    for model in "${models[@]}"; do
        echo ""
        echo -e "${YELLOW}[PROGRESS] Running model $((index + 1)) of ${#models[@]}: ${model_names[$index]}${NC}"
        
        run_model "$model"
        results[$model]=$?
        
        if [ ${results[$model]} -ne 0 ]; then
            ((failed_count++))
        fi
        
        ((index++))
    done
    
    # Generate execution summary
    echo ""
    echo "================================================================================"
    echo "  EXECUTION SUMMARY"
    echo "================================================================================"
    echo ""
    echo "  Model                          Status"
    echo "  ------------------------------  --------"
    
    index=0
    for model in "${models[@]}"; do
        if [ ${results[$model]} -eq 0 ]; then
            printf "  %-30s  ${GREEN}[PASS]${NC}\n" "${model_names[$index]}"
        else
            printf "  %-30s  ${RED}[FAIL]${NC}\n" "${model_names[$index]}"
        fi
        ((index++))
    done
    
    echo ""
    echo "  ------------------------------  --------"
    echo "  Total: ${#models[@]} models | Passed: $((${#models[@]} - failed_count)) | Failed: $failed_count"
    echo ""
    echo "================================================================================"
    
    if [ $failed_count -eq 0 ]; then
        echo -e "${GREEN}[SUCCESS] All models executed successfully!${NC}"
        return 0
    else
        echo -e "${RED}[WARNING] $failed_count model(s) failed during execution.${NC}"
        return 1
    fi
}

# ==============================================================================
# Main Entry Point
# ==============================================================================

# Display welcome banner
echo ""
echo "================================================================================"
echo "  Machine Learning Classification Project - Model Runner"
echo "  Version: 1.0 | Date: December 2025"
echo "================================================================================"

# Validate command line arguments
if [ $# -lt 1 ]; then
    echo -e "${RED}[ERROR] No model specified.${NC}"
    print_usage
    exit 1
fi

# Process command line argument
case $1 in
    all)
        run_all_models
        exit $?
        ;;
    help|-h|--help)
        print_usage
        exit 0
        ;;
    *)
        run_model "$1"
        exit $?
        ;;
esac

# ==============================================================================
# End of Script
# ==============================================================================
