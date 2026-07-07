require('dotenv').config();
const express = require('express');
const cors = require('cors');
const connectDB = require('../db');

const app = express();
app.use(cors());
app.use(express.json());

// Database connection middleware
app.use(async (req, res, next) => {
    try {
        await connectDB();
        next();
    } catch (err) {
        console.error('Database connection failed in serverless handler:', err);
        // We continue request handling; routes will handle MongoDB unavailable states gracefully or return errors.
        next();
    }
});

// Define API routes
app.use('/api/messages', require('../routes/messages'));
app.use('/api/comments', require('../routes/comments'));
app.use('/api/chat', require('../routes/chat'));

// Error handling middleware
app.use((err, req, res, next) => {
    console.error('Unhandled serverless error:', err);
    res.status(500).json({ success: false, error: 'Internal Server Error' });
});

module.exports = app;
