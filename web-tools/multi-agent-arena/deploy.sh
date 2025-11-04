#!/bin/bash

# Multi-Agent Arena Deployment Script
# Deploys backend to Railway and frontend to Vercel

set -e  # Exit on error

echo "🚀 Multi-Agent Arena Deployment"
echo "================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo -e "${RED}❌ Railway CLI not found${NC}"
    echo "Install with: npm install -g railway"
    exit 1
fi

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo -e "${RED}❌ Vercel CLI not found${NC}"
    echo "Install with: npm install -g vercel"
    exit 1
fi

# Check for required environment variables
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo -e "${YELLOW}⚠️  ANTHROPIC_API_KEY not set${NC}"
    read -p "Enter your Anthropic API key: " ANTHROPIC_API_KEY
fi

echo -e "${GREEN}✓ Prerequisites check passed${NC}"
echo ""

# Deploy Backend to Railway
echo "📦 Deploying backend to Railway..."
echo "==================================

"

cd backend

# Login to Railway if not already logged in
railway whoami || railway login

# Initialize project if needed
if [ ! -f ".railway" ]; then
    echo "Initializing Railway project..."
    railway init
fi

# Set environment variables
echo "Setting environment variables..."
railway variables set ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY"
railway variables set RATE_LIMIT_REQUESTS=3
railway variables set RATE_LIMIT_WINDOW=3600
railway variables set ENV=production
railway variables set LOG_LEVEL=INFO

# Deploy
echo "Deploying to Railway..."
railway up

# Get the backend URL
BACKEND_URL=$(railway status --json | jq -r '.deployments[0].url')
echo -e "${GREEN}✓ Backend deployed to: $BACKEND_URL${NC}"

cd ..

# Deploy Frontend to Vercel
echo ""
echo "📦 Deploying frontend to Vercel..."
echo "==================================="

cd frontend

# Set environment variable for backend URL
export NEXT_PUBLIC_API_URL="https://$BACKEND_URL"
export NEXT_PUBLIC_WS_URL="wss://$BACKEND_URL"

# Deploy to Vercel
echo "Deploying to Vercel..."
vercel --prod \
    -e NEXT_PUBLIC_API_URL="https://$BACKEND_URL" \
    -e NEXT_PUBLIC_WS_URL="wss://$BACKEND_URL"

# Get the frontend URL
FRONTEND_URL=$(vercel inspect --json | jq -r '.url')
echo -e "${GREEN}✓ Frontend deployed to: https://$FRONTEND_URL${NC}"

cd ..

# Update backend CORS
echo ""
echo "🔧 Updating backend CORS..."
cd backend
railway variables set CORS_ORIGINS="https://$FRONTEND_URL,http://localhost:3001"
echo -e "${GREEN}✓ CORS updated${NC}"

cd ..

# Success!
echo ""
echo "================================================"
echo -e "${GREEN}✅ Deployment complete!${NC}"
echo "================================================"
echo ""
echo "🌐 Frontend: https://$FRONTEND_URL"
echo "⚙️  Backend:  https://$BACKEND_URL"
echo ""
echo "📊 Next steps:"
echo "  1. Visit https://$FRONTEND_URL to test"
echo "  2. Set up monitoring: railway logs (backend) | vercel logs (frontend)"
echo "  3. Set up cost alerts in Railway dashboard"
echo ""
echo "🎉 Your Multi-Agent Arena is live!"
