# Fashion Recommender - React Frontend

A Pinterest-style UI for the Multimodal Recommendation System.

## Features

- User registration (auto-assigned ID)
- Pinterest-style masonry grid layout
- 4 interaction types: Like, Dislike, Save, Buy
- Cold-start recommendations (popular items)
- Personalized recommendations (after 5+ interactions)
- Real-time interaction tracking
- Responsive design

## Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm start
```

The app will run on http://localhost:3000

**Important:** Backend API must be running on http://localhost:8000

## Starting Both Frontend and Backend

```bash
# From project root
cd ml_service
../venv/Scripts/python.exe -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# In another terminal
cd frontend
npm start
```

## Components

- **App.js** - Main app with state management
- **UserLogin.jsx** - Registration/login form
- **ProductCard.jsx** - Individual product display with interactions
- **ProductGrid.jsx** - Pinterest masonry grid layout
- **api.js** - FastAPI client service

## User Flow

1. User enters name → Assigned user ID
2. Shows popular products (cold-start)
3. User interacts (like/save/buy)
4. After 5+ interactions → Personalized recommendations available
5. Click "Get My Recommendations" → See personalized items

## Technologies

- React 18
- Axios (API client)
- react-masonry-css (Grid layout)
- react-icons (Icons)
- react-toastify (Notifications)

## Available Scripts

### `npm start`
Runs the app in development mode on http://localhost:3000

### `npm run build`
Builds the app for production

### `npm test`
Launches the test runner
