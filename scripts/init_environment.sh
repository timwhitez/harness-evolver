#!/bin/bash
# Initialize the HarnessEvolver environment.
# Sets up Docker, Harbor, and Python dependencies.
# Run: source scripts/init_environment.sh

set -e

echo "=== HarnessEvolver Environment Setup ==="

# Check Docker
echo -n "Checking Docker... "
if command -v docker &> /dev/null; then
    echo "OK ($(docker --version))"
else
    echo "NOT FOUND — please install Docker: https://docs.docker.com/engine/install/"
fi

# Check Python
echo -n "Checking Python... "
if command -v python3 &> /dev/null; then
    echo "OK ($(python3 --version))"
else
    echo "NOT FOUND — please install Python 3.11+"
    exit 1
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -e ".[dev]" --break-system-packages 2>/dev/null || true

# Fix SSL in WSL
if [ -n "$WSL_DISTRO_NAME" ]; then
    export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
    grep -q "SSL_CERT_FILE" ~/.bashrc 2>/dev/null || \
        echo 'export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt' >> ~/.bashrc
    echo "WSL detected: SSL_CERT_FILE set to system CA bundle"
fi

# Check Harbor
echo -n "Checking Harbor... "
if command -v harbor &> /dev/null; then
    echo "OK (harbor $(harbor --version 2>/dev/null || echo 'installed'))"
else
    echo "NOT FOUND"
    echo "Install: pip install git+https://github.com/harbor-framework/harbor.git --break-system-packages"
    echo "See: https://github.com/harbor-framework/harbor"
fi

# Create required directories
echo "Creating runtime directories..."
mkdir -p trials/runs trials/summaries trials/regressions trials/diffs

# Bootstrap harness
echo "Bootstrapping harness config..."
python scripts/bootstrap_harness.py

echo ""
echo "=== Environment Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Set API keys: export ANTHROPIC_API_KEY=sk-..."
echo "  2. Run tests:    pytest tests/ -v"
echo "  3. Run a trial:  python scripts/run_trial.py --task <task_id>"
