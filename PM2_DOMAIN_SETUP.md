# Gene Web PM2 + Domain Setup Guide

## Overview

This updated setup uses **PM2** (Node process manager) to run all services and **domain-based routing** via Nginx.

**Architecture:**
```
gene.smarthsood.com          → port 3000   (Web Frontend)
api.gene.smarthsood.com      → ports 8000-8003 (Backend APIs)
auth.gene.smarthsood.com     → port 8004   (Auth Backend)
```

All services managed by PM2, auto-restart on failure, auto-start on reboot.

---

## Quick Start

### 1. Run Setup Script

```bash
# SSH into your EC2
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_IP>

# Run setup with your domain
sudo bash scripts/setup-vm-direct.sh gene.smarthsood.com
```

Or without specifying domain (defaults to `gene.smarthsood.com`):
```bash
sudo bash scripts/setup-vm-direct.sh
```

### 2. Configure Domain DNS

After setup completes, point your domain DNS records to your EC2 IP:

```
A record:  gene.smarthsood.com           → 1.2.3.4
A record:  api.gene.smarthsood.com       → 1.2.3.4
A record:  auth.gene.smarthsood.com      → 1.2.3.4
```

(Replace `1.2.3.4` with your actual EC2 public IP)

### 3. Configure Environment

```bash
sudo nano /opt/gene-web/.env
```

Update:
- `DATABASE_URL` - change password
- `JWT_SECRET` - generate with: `openssl rand -hex 32`
- `S3_ACCESS_KEY` & `S3_SECRET_KEY` (or use IAM role)

### 4. Start Services

```bash
gene-start
```

Check status:
```bash
gene-status
```

---

## PM2 Helper Commands

| Command | Purpose |
|---------|---------|
| `gene-start` | Start all services via PM2 |
| `gene-stop` | Stop all services |
| `gene-restart` | Restart all services |
| `gene-status` | Show PM2 status & resource usage |
| `gene-logs` | View logs for all services (Ctrl+C to exit) |
| `gene-reload` | Git pull + npm install + restart |

---

## Direct PM2 Commands

```bash
# Show all running processes
pm2 status

# Monitor processes in real-time
pm2 monit

# View logs for a specific service
pm2 logs model-backend
pm2 logs web-frontend

# View all logs
pm2 logs

# Restart specific service
pm2 restart model-backend

# Stop specific service
pm2 stop embedding-backend

# Delete a service from PM2
pm2 delete affinity-backend

# Show process details
pm2 show model-backend

# Save current state (for auto-restart on reboot)
pm2 save

# Restore state
pm2 resurrect
```

---

## Port Mapping

| Service | Port | Domain |
|---------|------|--------|
| Model Backend | 8000 | api.gene.smarthsood.com/models |
| Depmap Backend | 8001 | api.gene.smarthsood.com/associations |
| Embedding Backend | 8002 | api.gene.smarthsood.com/embeddings |
| Affinity Backend | 8003 | api.gene.smarthsood.com/affinity |
| Auth Backend | 8004 | auth.gene.smarthsood.com |
| Web Frontend | 3000 | gene.smarthsood.com |
| PostgreSQL | 5432 | localhost (internal) |
| Redis | 6379 | localhost (internal) |
| Nginx | 80 | (reverse proxy) |

---

## Accessing Services

### Web Frontend
```
http://gene.smarthsood.com
```

### Model Backend API
```
http://api.gene.smarthsood.com/models/docs
POST http://api.gene.smarthsood.com/models/predict
```

### Embedding Backend API
```
http://api.gene.smarthsood.com/embeddings/docs
POST http://api.gene.smarthsood.com/embeddings/compute
```

### Associations (Depmap) API
```
http://api.gene.smarthsood.com/associations/docs
GET http://api.gene.smarthsood.com/associations/search
```

### Affinity Backend API
```
http://api.gene.smarthsood.com/affinity/docs
POST http://api.gene.smarthsood.com/affinity/predict
```

### Auth Backend
```
http://auth.gene.smarthsood.com/docs
POST http://auth.gene.smarthsood.com/login
```

---

## Troubleshooting

### Services not starting?

Check PM2 logs:
```bash
pm2 logs
pm2 show model-backend  # Detailed info for specific service
```

Check environment variables:
```bash
cat /opt/gene-web/.env
```

Check if ports are already in use:
```bash
sudo netstat -tulpn | grep LISTEN
```

Kill process on port 8000:
```bash
sudo lsof -i :8000
sudo kill -9 <PID>
```

### Domain not resolving?

Check DNS:
```bash
nslookup gene.smarthsood.com
nslookup api.gene.smarthsood.com
```

Check Nginx is running:
```bash
sudo systemctl status nginx
sudo nginx -t  # Test config
```

Check Nginx error log:
```bash
sudo tail -50 /var/log/nginx/error.log
```

### Database connection errors?

Check PostgreSQL:
```bash
sudo systemctl status postgresql
sudo -u postgres psql -c "SELECT 1;"
```

Test connection:
```bash
psql -U gene_user -d gene_web -h localhost -c "SELECT 1;"
```

### Redis not accessible?

```bash
redis-cli ping
sudo systemctl status redis-server
sudo systemctl restart redis-server
```

### Nginx 502 Bad Gateway?

Usually means backend service is down:
```bash
pm2 status
pm2 logs
```

Check if port is listening:
```bash
curl localhost:8000  # Should work if service running
```

---

## Updating Code

To pull latest code and restart:

```bash
gene-reload
```

Or manually:
```bash
cd /opt/gene-web
git pull origin main
npm install
source venv/bin/activate
pip install -r requirements.txt
pm2 restart all
```

---

## HTTPS/SSL Setup (Optional)

Use Let's Encrypt with Certbot:

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d gene.smarthsood.com -d api.gene.smarthsood.com -d auth.gene.smarthsood.com

# Auto-renewal
sudo systemctl enable certbot.timer
sudo systemctl start certbot.timer
```

---

## Performance Monitoring

Real-time monitoring:
```bash
pm2 monit
```

Check system resources:
```bash
htop
free -h
df -h
```

---

## Auto-Start on Reboot

PM2 is already configured for auto-start:

```bash
# Verify
pm2 startup

# If needed, generate new startup script
pm2 startup systemd -u ubuntu --hp /home/ubuntu
pm2 save
```

After server reboot, check services:
```bash
pm2 status
pm2 logs
```

---

## Logs Location

**PM2 logs** (recommended):
```bash
pm2 logs
pm2 logs model-backend
~/.pm2/logs/  # Log files directory
```

**Nginx logs:**
```bash
/var/log/nginx/access.log
/var/log/nginx/error.log
```

**PostgreSQL logs:**
```bash
sudo -u postgres psql -l  # List databases
```

---

## Database Backup

Manual backup:
```bash
pg_dump -U gene_user gene_web > backup_$(date +%Y%m%d).sql
```

Automated backup (cron):
```bash
crontab -e
# Add: 0 2 * * * pg_dump -U gene_user gene_web > /opt/backups/gene_web_$(date +\%Y\%m\%d).sql
```

---

## Useful Links

- PM2 Docs: https://pm2.keymetrics.io/
- Nginx Docs: https://nginx.org/en/docs/
- PostgreSQL Docs: https://www.postgresql.org/docs/
- Let's Encrypt: https://letsencrypt.org/
