const mongoose = require('mongoose');

const certificateSchema = new mongoose.Schema({
    title: { type: String, required: true },
    description: { type: String, required: true },
    icon: { type: String, default: '🎓' },
    downloadUrl: { type: String, required: true },
    dateAdded: { type: Date, default: Date.now }
});

module.exports = mongoose.model('Certificate', certificateSchema);
