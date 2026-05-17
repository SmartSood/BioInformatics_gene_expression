# CI/CD Setup Guide

This guide explains how to configure the GitHub Actions workflow for automated builds and deployments.

## GitHub Actions Secrets Setup

You need to configure the following secrets in your GitHub repository settings under **Settings > Secrets and variables > Actions**.

### Docker Hub Credentials
- **DOCKER_USERNAME**: Your Docker Hub username
- **DOCKER_PASSWORD**: Your Docker Hub personal access token (not your password)

### AWS Credentials
- **AWS_ACCESS_KEY_ID**: AWS IAM user access key
- **AWS_SECRET_ACCESS_KEY**: AWS IAM user secret access key
- **AWS_REGION**: AWS region (e.g., `us-east-1`)
- **EKS_CLUSTER_NAME**: Your EKS cluster name

### Environment Variables

Create these as **Variables** (not secrets) for non-sensitive configuration:

- **DOCKER_REGISTRY**: `docker.io`
- **AWS_ACCOUNT_ID**: Your AWS account ID

## Prerequisites

### 1. Create Docker Hub Personal Access Token

1. Go to [Docker Hub](https://hub.docker.com/)
2. Click on your profile picture → Account Settings
3. Select **Security** → **Personal access tokens**
4. Click **Generate new token**
5. Give it a name (e.g., `github-actions`)
6. Grant read/write permissions
7. Copy the token

### 2. Create AWS IAM User for CI/CD

```bash
# Create IAM user
aws iam create-user --user-name github-actions

# Attach EKS and ECR policies
aws iam attach-user-policy \
  --user-name github-actions \
  --policy-arn arn:aws:iam::aws:policy/AmazonEKSFullAccess

aws iam attach-user-policy \
  --user-name github-actions \
  --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPowerUser

# Create access key
aws iam create-access-key --user-name github-actions
```

Save the `AccessKeyId` and `SecretAccessKey` as GitHub secrets.

### 3. Allow EKS Cluster Access

Make sure the IAM user has permission to access your EKS cluster:

```bash
# Update aws-auth ConfigMap in your EKS cluster
kubectl edit configmap aws-auth -n kube-system

# Add the IAM user mapping:
# - rolearn: arn:aws:iam::ACCOUNT_ID:user/github-actions
#   username: github-actions
#   groups:
#   - system:masters
```

## Workflow Triggers

The workflow is triggered by:

1. **Push to main branch** - Automatically builds and deploys
2. **Manual trigger** - Via "Run workflow" button in GitHub Actions tab
3. **Changes to specific paths**:
   - `apps/**`
   - `packages/**`
   - Dockerfiles
   - Workflow file itself

## Workflow Steps

### 1. Build and Push Stage
- Matrix builds all 6 Docker images in parallel
- Each image is tagged with commit SHA and `latest`
- Images are pushed to Docker Hub

### 2. Deploy to EKS Stage
- Updates Kubernetes manifests with new image tags
- Applies manifests to EKS cluster
- Waits for all deployments to be ready

### 3. Health Check Stage
- Verifies all services are responding
- Tests API endpoints

## Monitoring Deployments

### View Workflow Runs
1. Go to **Actions** tab in GitHub
2. Click on the latest run
3. View build logs for each service

### Monitor EKS Deployment
```bash
# Watch deployment progress
kubectl rollout status deployment/model-backend-api -n gene-web -w

# Check pod events
kubectl describe pod <pod-name> -n gene-web

# View logs
kubectl logs deployment/model-backend-api -n gene-web -f
```

## Troubleshooting

### Docker Build Fails
- Check that Dockerfile exists and is valid
- Ensure all dependencies are listed
- Check Docker Hub credentials are correct

### EKS Deployment Fails
- Verify AWS credentials are correct
- Check EKS cluster is running
- Ensure IAM user has required permissions
- Check kubectl can connect: `kubectl cluster-info`

### Image Pull Errors
- Verify image exists in Docker Hub
- Check image tag is correct
- Ensure EKS node has Docker Hub credentials (if using private registry)

### Deployment Timeout
- Check pod resource requests vs node availability
- Scale up EKS node group if needed
- Check if pods are in pending state: `kubectl describe pod <pod-name>`

## Manual Deployment

If you need to deploy without pushing code:

```bash
# Trigger workflow manually
# Go to Actions > Deploy Workflow > Run workflow

# Or deploy directly
kubectl apply -f k8s/
```

## Rollback

To rollback to a previous version:

```bash
# List deployment history
kubectl rollout history deployment/model-backend-api -n gene-web

# Rollback to previous version
kubectl rollout undo deployment/model-backend-api -n gene-web

# Rollback to specific revision
kubectl rollout undo deployment/model-backend-api -n gene-web --to-revision=2
```

## Security Best Practices

1. **Rotate secrets regularly**
   - Regenerate Docker Hub tokens
   - Rotate AWS access keys

2. **Use short-lived credentials**
   - Consider using AWS STS temporary credentials
   - Set token expiration for Docker Hub

3. **Limit IAM permissions**
   - Create dedicated IAM user for CI/CD
   - Use least-privilege access

4. **Never commit secrets**
   - Use GitHub Secrets for all sensitive data
   - Use `.gitignore` for local config files

5. **Enable branch protection**
   - Require status checks before merging
   - Require workflow to pass before deploying

## Next Steps

1. Set up all GitHub Secrets as described above
2. Configure your EKS cluster for IAM access
3. Push a change to trigger the first deployment
4. Monitor the workflow and verify deployment success
5. Set up branch protection rules
6. Configure notifications for failed deployments

## References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Hub Personal Access Tokens](https://docs.docker.com/docker-hub/access-tokens/)
- [AWS EKS Documentation](https://docs.aws.amazon.com/eks/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
