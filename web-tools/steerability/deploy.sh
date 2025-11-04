#!/bin/bash

# Steerability Dashboard Deployment Script
# Deploys backend to Railway and frontend to Vercel

set -e  # Exit on error

echo "🎛️  Steerability Dashboard Deployment"
echo "======================================"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check CLIs
if ! command -v railway &> /dev/null; then
    echo "Install Railway CLI: npm install -g railway"
    exit 1
fi

if ! command -v vercel &> /dev/null; then
    echo "Install Vercel CLI: npm install -g vercel"
    exit 1
fi

# Get API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
    read -p "Enter your Anthropic API key: " ANTHROPIC_API_KEY
fi

echo -e "${GREEN}✓ Prerequisites check passed${NC}"
echo ""

# Deploy Backend
echo "📦 Deploying backend to Railway..."
cd backend

railway whoami || railway login
[ ! -f ".railway" ] && railway init

railway variables set ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY"
railway variables set RATE_LIMIT_REQUESTS=5
railway variables set RATE_LIMIT_WINDOW=3600

railway up

BACKEND_URL=$(railway status --json | jq -r '.deployments[0].url')
echo -e "${GREEN}✓ Backend deployed to: $BACKEND_URL${NC}"

cd ..

# Deploy Frontend
echo ""
echo "📦 Deploying frontend to Vercel..."
cd frontend

vercel --prod \
    -e NEXT_PUBLIC_API_URL="https://$BACKEND_URL"

FRONTEND_URL=$(vercel inspect --json | jq -r '.url')
echo -e "${GREEN}✓ Frontend deployed to: https://$FRONTEND_URL${NC}"

cd ..

# Update CORS
cd backend
railway variables set CORS_ORIGINS="https://$FRONTEND_URL,http://localhost:3000"
cd ..

echo ""
echo "================================================"
echo -e "${GREEN}✅ Deployment complete!${NC}"
echo "================================================"
echo ""
echo "🌐 Frontend: https://$FRONTEND_URL"
echo "⚙️  Backend:  https://$BACKEND_URL"
echo ""
