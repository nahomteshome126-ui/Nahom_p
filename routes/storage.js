const express = require('express');
const router = express.Router();
const StoredFile = require('../models/StoredFile');

// Helper to check admin password from query or body
function isAdmin(req) {
    const adminPassword = process.env.ADMIN_PASSWORD;
    const providedPassword = req.query.password || req.body.password;
    return providedPassword && providedPassword === adminPassword;
}

// GET /api/storage - retrieve file list metadata (protected)
router.get('/', async (req, res) => {
    if (!isAdmin(req)) {
        return res.status(401).json({ success: false, error: 'Unauthorized' });
    }

    try {
        // Exclude the large 'data' field when listing files
        const files = await StoredFile.find({}).select('-data').sort({ uploadDate: -1 });
        res.json({ success: true, data: files });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// POST /api/storage - save a new file (protected)
router.post('/', async (req, res) => {
    if (!isAdmin(req)) {
        return res.status(401).json({ success: false, error: 'Unauthorized' });
    }

    try {
        const { filename, contentType, data } = req.body;
        if (!filename || !contentType || !data) {
            return res.status(400).json({ success: false, error: 'Missing required file data.' });
        }

        // Clean filename to prevent path traversal or weird issues
        const cleanFilename = filename.replace(/[^a-zA-Z0-9.\-_]/g, '_');

        // Check if filename already exists
        const existing = await StoredFile.findOne({ filename: cleanFilename });
        if (existing) {
            return res.status(400).json({ success: false, error: `File with name '${cleanFilename}' already exists.` });
        }

        // Calculate size from base64 string
        const buffer = Buffer.from(data, 'base64');
        const size = buffer.length;

        const newFile = new StoredFile({
            filename: cleanFilename,
            contentType,
            data,
            size
        });

        await newFile.save();
        res.status(201).json({ success: true, data: { _id: newFile._id, filename: newFile.filename, contentType: newFile.contentType, size: newFile.size, uploadDate: newFile.uploadDate } });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// DELETE /api/storage/:id - delete a file (protected)
router.delete('/:id', async (req, res) => {
    if (!isAdmin(req)) {
        return res.status(401).json({ success: false, error: 'Unauthorized' });
    }

    try {
        const file = await StoredFile.findByIdAndDelete(req.params.id);
        if (!file) {
            return res.status(404).json({ success: false, error: 'File not found' });
        }
        res.json({ success: true, message: 'File deleted successfully' });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// GET /api/storage/raw/:filename - serve the raw binary file content (public)
router.get('/raw/:filename', async (req, res) => {
    try {
        const file = await StoredFile.findOne({ filename: req.params.filename });
        if (!file) {
            return res.status(404).send('File not found');
        }

        // Decode base64 back to binary buffer
        const buffer = Buffer.from(file.data, 'base64');

        // Set Headers
        res.setHeader('Content-Type', file.contentType);
        res.setHeader('Content-Length', buffer.length);
        // Force view in browser if it's PDF/image/text, or download if other types
        const isInlineType = /pdf|image|text/i.test(file.contentType);
        res.setHeader('Content-Disposition', `${isInlineType ? 'inline' : 'attachment'}; filename="${file.filename}"`);
        res.setHeader('Cache-Control', 'public, max-age=86400'); // Cache for 1 day

        res.send(buffer);
    } catch (error) {
        res.status(500).send('Internal server error: ' + error.message);
    }
});

module.exports = router;
