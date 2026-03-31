const mongoose = require('mongoose');

const commentSchema = new mongoose.Schema({
    name: { type: String, required: true },
    email: { type: String, required: true },
    comment: { type: String, required: true },
    date: { type: Date, default: Date.now },
    approved: { type: Boolean, default: false } // admin must approve
});

module.exports = mongoose.model('Comment', commentSchema);