# Gene Web Direct VM Deployment - Quick Start Guide

## Overview
This guide sets up the entire Gene Web application stack directly on an Ubuntu EC2 instance without Kubernetes.

**Services running:**
- Model Backend API (port 8000)
- Embedding Backend API (port 8002)
- Depmap Backend API (port 8001)
- Affinity Backend API (port 8003)
- Auth Backend API (port 8004)
- Web Frontend (port 3000)
- PostgreSQL (port 5432)
- Redis (port 6379)
- Nginx Reverse Proxy (port 80)

---

## Step 1: Launch Ubuntu EC2 Instance

```bash
# Using AWS CLI
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \  # Ubuntu 22.04 LTS (us-east-1)
  --instance-type m6i.xlarge \
  --key-name your-key-pair \
  --security-group-ids sg-xxxxxxxx \
  --subnet-id subnet-xxxxxxxx \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=300,VolumeType=gp3}' \
  --iam-instance-profile 'Name=GeneWebEc2Role' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=gene-web-vm}]'
```

Or use AWS Console:
- AMI: Ubuntu Server 22.04 LTS
- Instance Type: m6i.xlarge
- Storage: 300 GB gp3
- Security Group: Allow SSH (22), HTTP (80), HTTPS (443)
- IAM Role: GeneWebEc2Role (with S3 access)

---

## Step 2: SSH into Instance

```bash
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_IP>

# Update and install curl (if not present)
sudo apt update
sudo apt install -y curl
```

---

## Step 3: Run Setup Script

Copy and paste this one-liner to download and run the setup script:

```bash
curl -fsSL https://raw.githubusercontent.com/YOUR_ORG/gene_web/main/scripts/setup-vm-direct.sh | sudo bash
```

Or if you have the script locally:

```bash
cd /path/to/gene_web
sudo bash scripts/setup-vm-direct.sh
```

**Setup takes approximately 15-20 minutes.** It will:
- ✅ Update system packages
- ✅ Install Docker, Node.js, Python, PostgreSQL, Redis
- ✅ Clone your repo
- ✅ Install dependencies
- ✅ Configure databases
- ✅ Create systemd services
- ✅ Configure Nginx reverse proxy
- ✅ Create helper commands

---

## Step 4: Configure Environment (CRITICAL)

After setup completes, edit the environment file:

```bash
sudo nano /opt/gene-web/.env
```

**Required changes:**
```bash
# 1. Change database password
DATABASE_URL=postgresql://gene_user:CHANGE_THIS_PASSWORD@localhost:5432/gene_web

# 2. Set secure JWT secret (generate with: openssl rand -hex 32)
JWT_SECRET=your_secure_random_string_here

# 3. S3 removed: use Neon DB only
# The deployment no longer uses S3 for artifact storage. Ensure your `DATABASE_URL`
# points to your Neon database and that `USE_S3=false` in the `.env` file.
# Example (already set in the script):
# USE_S3=false
# DATABASE_URL=postgresql://<user>:<password>@<neon_host>/<dbname>?sslmode=require

# 4. Update API URLs to your EC2 public IP:
MODEL_BACKEND_URL=http://<EC2_PUBLIC_IP>/api/models
EMBEDDING_BACKEND_URL=http://<EC2_PUBLIC_IP>/api/embeddings
# ... etc
```

Save with Ctrl+X, then Y, then Enter.

---

## Step 5: Start Services

After configuring `.env`, start all services:

```bash
gene-start
```

This starts:
1. PostgreSQL (already running from setup)
2. Redis (already running from setup)
3. Nginx (already running from setup)
4. All 6 backend/frontend services via systemd

Check status:

```bash
gene-status
```

Expected output:
```
● gene-model-backend.service - Gene Model Backend
     Loaded: loaded (/etc/systemd/system/gene-model-backend.service; enabled; vendor preset: enabled)
     Active: active (running) since ...
● gene-embedding-backend.service - Gene Embedding Backend
     Active: active (running) since ...
[... more services ...]
```

---

## Step 6: Verify Everything Works

```bash
# Get your EC2 public IP
curl http://169.254.169.254/latest/meta-data/public-ipv4

# Test from local machine
curl http://<EC2_PUBLIC_IP>/health

# Access web frontend
open http://<EC2_PUBLIC_IP>
# Or from command line:
curl http://<EC2_PUBLIC_IP>

# Test API endpoints
curl http://<EC2_PUBLIC_IP>/api/models/docs
curl http://<EC2_PUBLIC_IP>/api/embeddings/docs
curl http://<EC2_PUBLIC_IP>/api/associations/docs
```

---

## Useful Commands

### Service Management
```bash
# Start all
gene-start

# Stop all
gene-stop

# Check status
gene-status

# View logs for a service
gene-logs gene-model-backend
gene-logs gene-embedding-backend
gene-logs gene-web-frontend

# Manual systemd commands
systemctl start gene-model-backend
systemctl stop gene-model-backend
systemctl restart gene-model-backend
systemctl status gene-model-backend
systemctl enable gene-model-backend  # Auto-start on reboot

# Follow logs in real-time
journalctl -u gene-model-backend -f
journalctl -u gene-web-frontend -f

# Last 50 lines of service logs
journalctl -u gene-model-backend -n 50 --no-pager
```

### Database Management
```bash
# Connect to PostgreSQL
psql -U gene_user -d gene_web -h localhost

# Check Redis connection
redis-cli ping

# View Redis data
redis-cli
> KEYS *
> GET <key_name>
```

### Logs Location
```bash
# Systemd service logs (primary)
journalctl -u <service_name>

# Nginx error/access logs
/var/log/nginx/error.log
/var/log/nginx/access.log

# Alternative logs directory (if using manual start-all.sh)
/opt/gene-web/logs/
```

---

## Troubleshooting

### Services not starting?
```bash
# Check if ports are in use
sudo netstat -tulpn | grep LISTEN

# Kill process on specific port (e.g., 8000)
sudo lsof -i :8000
sudo kill -9 <PID>

# Restart specific service
systemctl restart gene-model-backend
```

### Database connection error?
```bash
# Verify PostgreSQL is running
systemctl status postgresql

# Test database connection
psql -U gene_user -d gene_web -h localhost -c "SELECT 1;"
```

### Redis not accessible?
```bash
# Check Redis
systemctl status redis-server
redis-cli ping  # Should return PONG

# Restart if needed
systemctl restart redis-server
```

### Nginx not routing traffic?
```bash
# Test nginx config
sudo nginx -t

# Check nginx status
systemctl status nginx

# View nginx error logs
sudo tail -50 /var/log/nginx/error.log

# Reload nginx
sudo systemctl reload nginx
```

### Service logs show errors?
```bash
# View last 100 lines of service logs
journalctl -u gene-model-backend -n 100 --no-pager

# Follow logs live
journalctl -u gene-model-backend -f

# Filter errors
journalctl -u gene-model-backend -p err --no-pager
```

---

## Updating Code

When you push updates to your repo:

```bash
cd /opt/gene-web
git pull origin main

# Reinstall Python deps (if requirements changed)
source venv/bin/activate
pip install -r requirements.txt

# Reinstall Node deps (if package.json changed)
npm install

# Restart affected services
systemctl restart gene-model-backend
systemctl restart gene-web-frontend
```

---

## Scaling & Performance Tips

1. **Increase service replicas** (not possible without Kubernetes, but you can:)
   - Run multiple instances of same service on different ports
   - Configure Nginx to load-balance between them
   - Use systemd socket activation for automatic scaling

2. **Monitor resource usage**
   ```bash
   htop
   free -h
   df -h
   ```

3. **Enable swap** (if you run out of RAM)
   ```bash
   sudo fallocate -l 8G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

4. **Increase file descriptors** (for high-load)
   ```bash
   ulimit -n 65536
   ```

---

## Security Notes

1. **Change all default passwords** in `.env` file
2. **Use AWS Security Groups** to restrict access (not open to 0.0.0.0)
3. **Enable HTTPS** with Let's Encrypt:
   ```bash
   sudo apt install certbot python3-certbot-nginx
   sudo certbot --nginx -d your-domain.com
   ```
4. **Set strong JWT secret** (use: `openssl rand -hex 32`)
5. **Use AWS IAM roles** for S3 access (not static keys)
6. **Rotate credentials regularly**

---

## Stopping the Stack

To shut down everything gracefully:

```bash
gene-stop
```

To reboot and auto-start on next boot:

```bash
sudo reboot
# Services will auto-start (systemctl enable was set)
```

---

## Next Steps

1. ✅ Configure `.env` with your credentials
2. ✅ Start services with `gene-start`
3. ✅ Test endpoints
4. ✅ Set up domain + HTTPS (if going to production)
5. ✅ Configure backups for PostgreSQL/uploads
6. ✅ Monitor with CloudWatch/Datadog
