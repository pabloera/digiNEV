#!/bin/bash

# Quick Setup Script for TDD Development
# Sets up the test environment and dependencies

echo "🚀 TDD Development Environment Setup"
echo "=================================="

# Navigate to project directory
cd /Users/pabloalmada/development/project/dataanalysis-bolsonarismo

# Activate virtual environment
echo "📦 Activating virtual environment..."
source .venv/bin/activate

# Install missing dependencies for TDD
echo "📥 Installing TDD dependencies..."
pip install pytest pytest-cov pytest-html

# Install core dependencies identified by tests
echo "📥 Installing core dependencies..."
pip install anthropic voyageai

# Install performance dependencies  
echo "📥 Installing performance dependencies..."
pip install lz4 psutil

# Install additional NLP dependencies
echo "📥 Installing NLP dependencies..."
pip install textblob vaderSentiment

# Run quick test to verify setup
echo "🧪 Running quick test verification..."
python tests/run_tests.py quick

echo ""
echo "✅ TDD Environment Setup Complete!"
echo "🎯 Ready for Phase 3: Implementation"
echo ""
echo "Next steps:"
echo "1. Fix test data issues: python tests/fix_test_data.py"
echo "2. Begin implementation: Start with core classes"
echo "3. Run tests iteratively: python tests/run_tests.py"
echo ""
echo "Happy coding! 🚀"
