const mongoose = require('mongoose');

let cachedConnection = null;

async function connectDB() {
    if (cachedConnection) {
        return cachedConnection;
    }

    if (!process.env.MONGODB_URI) {
        console.warn('⚠️ MONGODB_URI is not set. Database features will be unavailable.');
        return null;
    }

    try {
        // Connect to MongoDB
        const conn = await mongoose.connect(process.env.MONGODB_URI, {
            bufferCommands: false,
        });
        cachedConnection = conn;
        console.log('✅ Connected to MongoDB');
        return conn;
    } catch (error) {
        console.error('❌ MongoDB connection error:', error);
        throw error;
    }
}

module.exports = connectDB;
