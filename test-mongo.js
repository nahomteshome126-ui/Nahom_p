const mongoose = require('mongoose');

(async () => {
  const uri = process.env.MONGODB_URI;
  if (!uri) {
    console.error('❌ MONGODB_URI is not set. Export it before running this script.');
    process.exit(1);
  }

  try {
    const conn = await mongoose.connect(uri, { bufferCommands: false });
    console.log('✅ Connected to MongoDB (test-mongo.js)');
    await mongoose.disconnect();
    process.exit(0);
  } catch (err) {
    console.error('❌ Connection failed:', err.message || err);
    process.exit(1);
  }
})();
