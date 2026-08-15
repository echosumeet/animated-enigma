# MARKET POKER — CLOUD & LOCAL DEPLOYMENT

## 1. Local Development
```bash
# Clone repository
git clone <repository>
cd market-poker

# Install dependencies
npm install

# Run dev server on port 3000
npm run dev
```

## 2. Production Build
```bash
# Typecheck & build static bundle
npm run build

# Preview production build
npm run preview
```

## 3. Container & Cloud Deployment
- **Port:** 3000
- **Host:** 0.0.0.0
- **Environment:** Single-Page Application / Express Server compatible with Google Cloud Run, Vercel, AWS ECS, Railway, or Render.
