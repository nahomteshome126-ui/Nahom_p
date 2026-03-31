const express = require('express');
const router = express.Router();
const Comment = require('../models/Comment');

// Helper to check admin password (same as before)
function isAdmin(req) {
    const adminPassword = process.env.ADMIN_PASSWORD;
    const providedPassword = req.query.password;
    return providedPassword && providedPassword === adminPassword;
}

// GET /api/comments – public: only approved comments
router.get('/', async (req, res) => {
    try {
        const comments = await Comment.find({ approved: true }).sort({ date: -1 });
        res.json({ success: true, data: comments });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// POST /api/comments – public: submit a new comment (unapproved)
router.post('/', async (req, res) => {
    try {
        const { name, email, comment } = req.body;
        const newComment = new Comment({ name, email, comment });
        await newComment.save();
        res.status(201).json({ success: true, data: newComment });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// GET /api/comments/all – admin: get all comments (with optional filter)
router.get('/all', async (req, res) => {
    if (!isAdmin(req)) {
        return res.status(401).json({ success: false, error: 'Unauthorized' });
    }
    try {
        const comments = await Comment.find().sort({ date: -1 });
        res.json({ success: true, data: comments });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// PATCH /api/comments/:id/approve – admin: toggle approved status
router.patch('/:id/approve', async (req, res) => {
    if (!isAdmin(req)) {
        return res.status(401).json({ success: false, error: 'Unauthorized' });
    }
    try {
        const comment = await Comment.findById(req.params.id);
        if (!comment) {
            return res.status(404).json({ success: false, error: 'Comment not found' });
        }
        comment.approved = !comment.approved; // toggle
        await comment.save();
        res.json({ success: true, data: comment });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// DELETE /api/comments/:id – admin: delete comment permanently (optional)
router.delete('/:id', async (req, res) => {
    if (!isAdmin(req)) {
        return res.status(401).json({ success: false, error: 'Unauthorized' });
    }
    try {
        await Comment.findByIdAndDelete(req.params.id);
        res.json({ success: true });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

module.exports = router;