#!/bin/bash
# ==============================================================================
# 1-Click Deploy Script: Hermes Autonomous Agent on DigitalOcean Droplet
# Configured for DigitalOcean Serverless Inference (https://inference.do-ai.run/v1)
# ==============================================================================

set -euo pipefail

echo "🌊 Starting 1-Click Hermes Agent Deployment on DigitalOcean..."

# 1. Update packages & install dependencies
apt-get update -y
apt-get install -y python3-pip python3-venv git curl jq

# 2. Configure DigitalOcean Serverless Inference Environment
DO_MODEL_ACCESS_KEY="${DO_MODEL_ACCESS_KEY:-}"
INFERENCE_MODEL="${INFERENCE_MODEL:-llama-3.3-70b-instruct}"
DO_INFERENCE_ENDPOINT="https://inference.do-ai.run/v1"

cat << EOF > /etc/environment
DO_INFERENCE_ENDPOINT="${DO_INFERENCE_ENDPOINT}"
MODEL_ACCESS_KEY="${DO_MODEL_ACCESS_KEY}"
INFERENCE_MODEL="${INFERENCE_MODEL}"
OPENAI_BASE_URL="${DO_INFERENCE_ENDPOINT}"
OPENAI_API_KEY="${DO_MODEL_ACCESS_KEY}"
EOF

# 3. Create hermes system user & directories
useradd -m -s /bin/bash hermes || true
mkdir -p /opt/hermes /home/hermes/.hermes

# 4. Set up Hermes Agent Virtual Environment
python3 -m venv /opt/hermes/venv
/opt/hermes/venv/bin/pip install --upgrade pip
/opt/hermes/venv/bin/pip install hermes-agent

# 5. Bind Hermes directly to DigitalOcean Serverless Inference
cat << EOF > /home/hermes/.hermes/config.yaml
version: 1
provider: openai-compatible
base_url: ${DO_INFERENCE_ENDPOINT}
api_key: ${DO_MODEL_ACCESS_KEY}
model: ${INFERENCE_MODEL}
system_prompt: "You are the 24/7 RxGuard Medication Safety Copilot running on DigitalOcean. You monitor high-risk medication alerts, answer patient questions about dosages and food interactions, and follow up with caregivers."
tools:
  - web_search
  - terminal
  - medical_calculator
EOF

chown -R hermes:hermes /opt/hermes /home/hermes/.hermes

# 6. Create Systemd Service for 24/7 Background Persistence
cat << EOF > /etc/systemd/system/hermes-agent.service
[Unit]
Description=Hermes 24/7 Autonomous Caregiver Agent on DigitalOcean
After=network.target

[Service]
Type=simple
User=hermes
WorkingDirectory=/home/hermes
ExecStart=/opt/hermes/venv/bin/hermes gateway start
Restart=always
RestartSec=5
EnvironmentFile=/etc/environment

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable hermes-agent || true

echo "✅ Hermes Agent is successfully installed and paired with DigitalOcean Serverless Inference!"
echo "👉 To check status: systemctl status hermes-agent"
echo "👉 Interactive session: su - hermes -c '/opt/hermes/venv/bin/hermes doctor'"
