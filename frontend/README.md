# Frontend - Cardamom Leaf Disease Detection

React TypeScript frontend for the cardamom leaf disease detection system.

## Features

- **Image Upload**: Drag and drop or select image files
- **Real-time Preview**: See uploaded images before analysis
- **Disease Prediction**: Get instant classification results
- **Confidence Visualization**: View prediction confidence with progress bars
- **Grad-CAM Heatmap**: Visual explanation of model predictions
- **Error Handling**: User-friendly error messages
- **Responsive Design**: Works on desktop and mobile devices
- **Modern UI**: Beautiful gradient design with smooth animations

## Tech Stack

- **React 19** - UI library
- **TypeScript** - Type safety
- **Vite** - Fast build tool and dev server
- **Axios** - HTTP client for API calls
- **CSS3** - Modern styling with gradients and animations

## Project Structure

```
frontend/
├── src/
│   ├── main.tsx              # React entry point
│   ├── App.tsx               # Main application component
│   ├── App.css               # Application styles
│   ├── index.css             # Global styles
│   └── api/
│       └── client.ts         # API client for backend communication
├── public/                   # Static assets
├── index.html               # HTML template
├── package.json             # Dependencies and scripts
├── tsconfig.json            # TypeScript configuration
├── tsconfig.node.json       # TypeScript config for Node
└── vite.config.ts           # Vite configuration
```

## Setup

### Prerequisites

- Node.js 18 or higher
- npm or yarn

### Installation

1. Install dependencies:
```bash
cd frontend
npm install
```

## Running the Development Server

Start the development server:
```bash
npm run dev
```

The application will be available at: http://localhost:5173

## Building for Production

Build the application:
```bash
npm run build
```

The built files will be in the `dist/` directory.

Preview the production build:
```bash
npm run preview
```

## API Configuration

The frontend communicates with the backend API running at `http://localhost:8000`.

To change the API URL, edit `src/api/client.ts`:

```typescript
const API_BASE_URL = 'http://localhost:8000';
```

## Features Overview

### 1. Image Upload
- Click "Choose Image" button to select an image file
- Supports all common image formats (JPEG, PNG, etc.)
- Validates file type before upload

### 2. Image Preview
- Shows uploaded image before analysis
- Maintains aspect ratio
- Responsive sizing

### 3. Analysis
- Click "Analyze" button to send image to backend
- Shows loading spinner during processing
- Timeout set to 30 seconds

### 4. Results Display
- **Disease Class**: Predicted disease category
  - Colletotrichum Blight
  - Phyllosticta Leaf Spot
  - Healthy
- **Confidence**: Prediction confidence as percentage (0-100%)
- **Confidence Bar**: Visual representation of confidence
- **Grad-CAM Heatmap**: Overlayed visualization showing important regions

### 5. Error Handling
Handles various error scenarios:
- Invalid file type
- Backend connection issues
- Request timeouts
- Server errors

### 6. Reset Functionality
- Clear all state and start over
- Cleans up memory (URL.revokeObjectURL)

## Component Structure

### App.tsx
Main application component with:
- File selection state management
- Image preview with URL.createObjectURL
- API call handling
- Loading and error states
- Results display

### API Client (api/client.ts)
Axios-based HTTP client with:
- TypeScript interfaces for type safety
- Health check endpoint
- Prediction endpoint with FormData
- 30-second timeout
- Error handling

## Styling

The application uses modern CSS with:
- **Gradient Background**: Purple/blue gradient
- **Card Design**: White card with shadow
- **Smooth Animations**: Fade-in effects and transitions
- **Responsive Layout**: Adapts to different screen sizes
- **Loading States**: Spinner animations
- **Interactive Elements**: Hover effects and button states

### Color Scheme
- Primary Gradient: `#667eea` to `#764ba2`
- Background: White cards on gradient background
- Text: Dark gray for readability
- Accents: Purple/blue from gradient

## Development

### Code Quality

Lint the code:
```bash
npm run lint
```

### Type Checking

TypeScript provides compile-time type checking. Run:
```bash
npm run build
```

## Troubleshooting

### Cannot Connect to Backend

If you see "Cannot connect to server" error:
1. Make sure backend is running on port 8000
2. Check that CORS is properly configured
3. Verify the API URL in `src/api/client.ts`

### Build Errors

If you encounter build errors:
1. Delete `node_modules` and reinstall:
   ```bash
   rm -rf node_modules
   npm install
   ```
2. Clear Vite cache:
   ```bash
   rm -rf node_modules/.vite
   ```

### Port Already in Use

If port 5173 is in use, Vite will automatically try the next available port.

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)

## License

MIT
