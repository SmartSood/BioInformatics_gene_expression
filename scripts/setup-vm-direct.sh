#!/bin/bash
set -e

# Gene Web Direct VM Setup Script with PM2 & Domain Routing
# This script sets up all dependencies and services on Ubuntu EC2
# Services run on different ports managed by PM2, routed via Nginx to domain
# Usage: sudo bash setup-vm-direct.sh

DOMAIN=${1:-gene.smarthsood.com}
echo "🚀 Starting Gene Web VM Setup..."
echo "📍 Domain: $DOMAIN"

# ============================================================================
# 1. Update system packages
# ============================================================================
echo "📦 Updating system packages..."
apt-get update
apt-get upgrade -y

# ============================================================================
# 2. Install core dependencies
# ============================================================================
echo "📦 Installing core dependencies..."
apt-get install -y \
  curl \
  wget \
  git \
  vim \
  htop \
  build-essential \
  libssl-dev \
  libffi-dev \
  python3-dev \
  python3-pip \
  python3-venv \
  nodejs \
  npm \
  redis-server \
  postgresql \
  postgresql-contrib \
  nginx

# ============================================================================
# 3. Install PM2 globally
# ============================================================================
echo "📦 Installing PM2 (process manager)..."
npm install -g pm2
pm2 completion install

# ============================================================================
# 4. Create app directory and clone repo
# ============================================================================
echo "📁 Setting up application directory..."
APP_DIR="/opt/gene-web"
mkdir -p "$APP_DIR"
cd "$APP_DIR"

# Only clone if not already cloned
if [ ! -d "$APP_DIR/.git" ]; then
  echo "🔄 Cloning gene_web repository..."
  # Replace with your actual repo URL
  git clone https://github.com/YOUR_ORG/gene_web.git .
else
  echo "✓ Repository already cloned"
  git pull origin main
fi

# ============================================================================
# 5. Setup Python virtual environment
# ============================================================================
echo "🐍 Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# If individual backend requirements exist
for req_file in apps/*/requirements.txt; do
  if [ -f "$req_file" ]; then
    echo "📦 Installing $(dirname $req_file) requirements..."
    pip install -r "$req_file" || true
  fi
done

# ============================================================================
# 6. Setup Node.js dependencies
# ============================================================================
echo "📦 Installing Node dependencies..."
npm install

# Install workspace dependencies
npm install -w apps/auth_backend
npm install -w apps/web

# ============================================================================
# 7. Configure PostgreSQL
# ============================================================================
echo "🗄️  Configuring PostgreSQL..."
sudo -u postgres psql <<EOF
CREATE DATABASE gene_web;
CREATE USER gene_user WITH PASSWORD 'gene_password_change_me';
GRANT ALL PRIVILEGES ON DATABASE gene_web TO gene_user;
ALTER ROLE gene_user SET client_encoding TO 'utf8';
ALTER ROLE gene_user SET default_transaction_isolation TO 'read committed';
ALTER ROLE gene_user SET default_transaction_deferrable TO on;
ALTER ROLE gene_user SET default_transaction_level TO 'read committed';
ALTER ROLE gene_user SET timezone TO 'UTC';
EOF

# Start PostgreSQL
systemctl start postgresql
systemctl enable postgresql

# ============================================================================
# 8. Configure Redis
# ============================================================================
echo "🔴 Configuring Redis..."
systemctl start redis-server
systemctl enable redis-server

# ============================================================================
# 9. Create environment file
# ============================================================================
echo "⚙️  Creating environment configuration..."
cat > "$APP_DIR/.env" <<'EOF'
# Database
DATABASE_URL=postgresql://gene_user:gene_password_change_me@localhost:5432/gene_web

# Redis
REDIS_URL=redis://localhost:6379

# S3 Configuration
USE_S3=true
S3_BUCKET=gene-web-data
S3_REGION=us-east-1
# Use AWS IAM role or set these:
# S3_ACCESS_KEY=your_access_key
# S3_SECRET_KEY=your_secret_key

# JWT
JWT_SECRET=change_me_to_a_secure_random_string_at_least_32_chars_long

# APIs
MODEL_BACKEND_URL=http://localhost:8000
EMBEDDING_BACKEND_URL=http://localhost:8002
DEPMAP_BACKEND_URL=http://localhost:8001
AFFINITY_BACKEND_URL=http://localhost:8003
AUTH_BACKEND_URL=http://localhost:8004

# Frontend
NEXT_PUBLIC_API_URL=http://localhost/api
NEXT_PUBLIC_AUTH_URL=http://localhost/auth

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# Logging
LOG_LEVEL=info
EOF

chmod 600 "$APP_DIR/.env"

# ============================================================================
# 9. Create PM2 Ecosystem Config
# ============================================================================
echo "⚙️  Creating PM2 ecosystem configuration..."
cat > "$APP_DIR/ecosystem.config.js" <<'EOF'
module.exports = {
  apps: [
    {
      name: 'model-backend',
      script: 'apps/model_backend/server.py',
      interpreter: './venv/bin/python',
      args: '--host 0.0.0.0 --port 8000',
      exec_mode: 'fork',
      instances: 1,
      autorestart: true,
      watch: false,
      env: {
        NODE_ENV: 'production'
      }
    },
    {
      name: 'embedding-backend',
      script: 'apps/embedding_backend/server.py',
      interpreter: './venv/bin/python',
      args: '--host 0.0.0.0 --port 8002',
      exec_mode: 'fork',
      instances: 1,
      autorestart: true,
      watch: false,
    },
    {
      name: 'depmap-backend',
      script: 'apps/depmap_backend/server.py',
      interpreter: './venv/bin/python',
      args: '--host 0.0.0.0 --port 8001',
      exec_mode: 'fork',
      instances: 1,
      autorestart: true,
      watch: false,
    },
    {
      name: 'affinity-backend',
      script: 'apps/affinity_backend/server.py',
      interpreter: './venv/bin/python',
      args: '--host 0.0.0.0 --port 8003',
      exec_mode: 'fork',
      instances: 1,
      autorestart: true,
      watch: false,
    },
    {
      name: 'auth-backend',
      script: 'npm',
      args: 'run start --workspace apps/auth_backend',
      instances: 1,
      autorestart: true,
      watch: false,
    },
    {
      name: 'web-frontend',
      script: 'npm',
      args: 'run start --workspace apps/web',
      instances: 1,
      autorestart: true,
      watch: false,
    },
  ]
};
EOF

# ============================================================================
# 10. Configure Nginx Reverse Proxy with Domain Routing
# ============================================================================
echo "🌐 Configuring Nginx reverse proxy for $DOMAIN..."
cat > /etc/nginx/sites-available/gene-web <<EOF
upstream model_backend {
    server localhost:8000;
}

upstream embedding_backend {
    server localhost:8002;
}

upstream depmap_backend {
    server localhost:8001;
}

upstream affinity_backend {
    server localhost:8003;
}

upstream auth_backend {
    server localhost:8004;
}

upstream web_frontend {
    server localhost:3000;
}

# Redirect HTTP to HTTPS (optional - remove if no SSL)
# server {
#     listen 80;
#     server_name $DOMAIN api.$DOMAIN auth.$DOMAIN;
#     return 301 https://\$host\$request_uri;
# }

# Main domain
server {
    listen 80;
    server_name $DOMAIN;
    client_max_body_size 100M;

    location / {
        proxy_pass http://web_frontend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_cache_bypass \$http_upgrade;
    }

    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}

# API subdomain - routes to all backend APIs
server {
    listen 80;
    server_name api.$DOMAIN;
    client_max_body_size 100M;

    # Model backend routes
    location /models/ {
        proxy_pass http://model_backend/;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    # Embedding backend routes
    location /embeddings/ {
        proxy_pass http://embedding_backend/;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    # Depmap/Associations routes
    location /associations/ {
        proxy_pass http://depmap_backend/;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    # Affinity routes
    location /affinity/ {
        proxy_pass http://affinity_backend/;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}

# Auth subdomain
server {
    listen 80;
    server_name auth.$DOMAIN;
    client_max_body_size 100M;

    location / {
        proxy_pass http://auth_backend;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

# Enable site
ln -sf /etc/nginx/sites-available/gene-web /etc/nginx/sites-enabled/gene-web
rm -f /etc/nginx/sites-enabled/default

# Test nginx config
nginx -t

# Start nginx
systemctl start nginx
systemctl enable nginx

# ============================================================================
# 11. Create PM2 Helper Commands
# ============================================================================
echo "📝 Creating PM2 helper commands..."

cat > /usr/local/bin/gene-start <<'EOF'
#!/bin/bash
cd /opt/gene-web
source .env
pm2 start ecosystem.config.js
pm2 save
echo "✅ All services started via PM2"
echo "Access: http://gene.smarthsood.com (or your configured domain)"
EOF
chmod +x /usr/local/bin/gene-start

cat > /usr/local/bin/gene-stop <<'EOF'
#!/bin/bash
pm2 stop all
echo "✅ All services stopped"
EOF
chmod +x /usr/local/bin/gene-stop

cat > /usr/local/bin/gene-restart <<'EOF'
#!/bin/bash
pm2 restart all
echo "✅ All services restarted"
EOF
chmod +x /usr/local/bin/gene-restart

cat > /usr/local/bin/gene-status <<'EOF'
#!/bin/bash
echo "🔍 Gene Web Service Status (PM2):"
pm2 status
echo ""
echo "📊 Resource Usage:"
pm2 monit
EOF
chmod +x /usr/local/bin/gene-status

cat > /usr/local/bin/gene-logs <<'EOF'
#!/bin/bash
SERVICE=${1:-all}
if [ "$SERVICE" == "all" ]; then
  pm2 logs
else
  pm2 logs $SERVICE
fi
EOF
chmod +x /usr/local/bin/gene-logs

cat > /usr/local/bin/gene-reload <<'EOF'
#!/bin/bash
cd /opt/gene-web
git pull origin main
npm install
source venv/bin/activate
pip install -r requirements.txt
pm2 restart all
echo "✅ Code updated and services restarted"
EOF
chmod +x /usr/local/bin/gene-reload

# ============================================================================
# 12. Setup PM2 Auto-Start on Reboot
# ============================================================================
echo "🔄 Setting up PM2 auto-start on reboot..."
pm2 startup systemd -u ubuntu --hp /home/ubuntu
pm2 save

# ============================================================================
# 13. Summary & Instructions
# ============================================================================
echo ""
echo "✅ Gene Web VM Setup Complete!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 IMPORTANT: Configuration Steps Required"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1. Edit environment file and update credentials:"
echo "   sudo nano /opt/gene-web/.env"
echo ""
echo "   Required changes:"
echo "   - Change DATABASE password"
echo "   - Set JWT_SECRET to a secure random string"
echo "   - Configure S3 credentials (S3_ACCESS_KEY, S3_SECRET_KEY) or use IAM role"
echo "   - Update domain URLs if using different domain than gene.smarthsood.com"
echo ""
echo "2. Configure your domain DNS to point to this server:"
echo "   A record:     gene.smarthsood.com     → <EC2_PUBLIC_IP>"
echo "   A record:     api.gene.smarthsood.com → <EC2_PUBLIC_IP>"
echo "   A record:     auth.gene.smarthsood.com → <EC2_PUBLIC_IP>"
echo ""
echo "3. Start services using PM2:"
echo "   gene-start          # Start all services"
echo "   gene-stop           # Stop all services"
echo "   gene-restart        # Restart all services"
echo "   gene-status         # Check service status"
echo "   gene-logs           # View all logs (Ctrl+C to exit)"
echo "   gene-reload         # Pull latest code and restart"
echo ""
echo "4. Access the application:"
echo "   Web:  http://gene.smarthsood.com"
echo "   APIs: http://api.gene.smarthsood.com/{models,embeddings,associations,affinity}/"
echo "   Auth: http://auth.gene.smarthsood.com"
echo ""
echo "5. Monitor in real-time:"
echo "   pm2 monit"
echo "   pm2 logs"
echo "   pm2 status"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
