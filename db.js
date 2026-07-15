const mongoose = require('mongoose');

let cachedPromise = null;

async function connectDB() {
    if (mongoose.connection.readyState === 1) {
        return mongoose.connection;
    }

    if (!process.env.MONGODB_URI) {
        console.warn('⚠️ MONGODB_URI is not set. Database features will be unavailable.');
        return null;
    }

    if (!cachedPromise) {
        console.log('🔌 Initiating new MongoDB connection...');
        cachedPromise = mongoose.connect(process.env.MONGODB_URI, {
            bufferCommands: false,
        }).then((m) => {
            console.log('✅ Connected to MongoDB');
            return m;
        }).catch((error) => {
            console.error('❌ MongoDB connection error:', error);
            cachedPromise = null; // Reset cached promise on failure to allow retry
            throw error;
        });
    }

    return cachedPromise;
}

module.exports = connectDB;

