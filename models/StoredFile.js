const mongoose = require('mongoose');

const storedFileSchema = new mongoose.Schema({
    filename: { type: String, required: true, unique: true },
    contentType: { type: String, required: true },
    data: { type: String, required: true }, // Base64 encoded file content
    size: { type: Number, required: true }, // size in bytes
    uploadDate: { type: Date, default: Date.now }
});

module.exports = mongoose.model('StoredFile', storedFileSchema);
