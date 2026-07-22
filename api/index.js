require('dotenv').config();
const express = require('express');
const path = require('path');
const cors = require('cors');
const connectDB = require('../db');

const app = express();
app.use(cors());
app.use(express.json());

// Serve static files from dist/ directory (for production builds)
const distPath = path.join(__dirname, '../dist');
app.use(express.static(distPath));

// Database connection middleware
app.use(async (req, res, next) => {
    // Bypassing DB check for chat route so it stays functional without database configuration
    if (req.path.startsWith('/chat') || req.path === '/api/chat') {
        console.log('⏭️  Skipping DB check for chat endpoint');
        return next();
    }

    try {
        const conn = await connectDB();
        if (!conn) {
            return res.status(500).json({ 
                success: false, 
                error: 'Database configuration missing. Please configure MONGODB_URI in your Vercel Project Dashboard.' 
            });
        }
        next();
    } catch (err) {
        console.error('Database connection failed in serverless handler:', err);
        return res.status(500).json({ 
            success: false, 
            error: 'Database connection failed. Please verify that MONGODB_URI is correct and that MongoDB Atlas allows access from all IPs (0.0.0.0/0).' 
        });
    }
});

// Define API routes
app.use('/api/messages', require('../routes/messages'));
app.use('/api/comments', require('../routes/comments'));
app.use('/api/certificates', require('../routes/certificates'));
app.use('/api/chat', require('../routes/chat'));
app.use('/api/storage', require('../routes/storage'));

// Error handling middleware
app.use((err, req, res, next) => {
    console.error('Unhandled serverless error:', err);
    res.status(500).json({ success: false, error: 'Internal Server Error' });
});

module.exports = app;
