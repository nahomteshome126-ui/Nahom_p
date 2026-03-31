const express = require('express');
const router = express.Router();
const Message = require('../models/Message');

// Helper to check admin password
function isAdmin(req) {
    const adminPassword = process.env.ADMIN_PASSWORD;
    const providedPassword = req.query.password;
    return providedPassword && providedPassword === adminPassword;
}

// GET /api/messages – retrieve messages (protected)
// If includeArchived=true and password correct, returns all; else only non-archived
router.get('/', async (req, res) => {
    const includeArchived = req.query.includeArchived === 'true';
    const admin = isAdmin(req);

    // Build filter: if not admin or not including archived, only show non-archived
    let filter = {};
    if (!admin || !includeArchived) {
        filter.archived = false;
    }

    try {
        const messages = await Message.find(filter).sort({ date: -1 });
        res.json({ success: true, data: messages });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// POST /api/messages – save a new message (public)
router.post('/', async (req, res) => {
    try {
        const { name, email, message } = req.body;
        const newMessage = new Message({ name, email, message });
        await newMessage.save();
        res.status(201).json({ success: true, data: newMessage });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// PATCH /api/messages/:id/archive – toggle archived status (protected)
router.patch('/:id/archive', async (req, res) => {
    if (!isAdmin(req)) {
        return res.status(401).json({ success: false, error: 'Unauthorized' });
    }

    try {
        const message = await Message.findById(req.params.id);
        if (!message) {
            return res.status(404).json({ success: false, error: 'Message not found' });
        }

        message.archived = !message.archived;
        await message.save();
        res.json({ success: true, data: message });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

module.exports = router;