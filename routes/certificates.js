const express = require('express');
const router = express.Router();
const Certificate = require('../models/Certificate');

// Helper to check admin password
function isAdmin(req) {
    const adminPassword = process.env.ADMIN_PASSWORD;
    const providedPassword = req.query.password || req.body.password;
    return providedPassword && providedPassword === adminPassword;
}

// GET /api/certificates - Public (fetches all certificates; seeds defaults if empty)
router.get('/', async (req, res) => {
    try {
        let certs = await Certificate.find().sort({ dateAdded: -1 });
        
        // Seed default certificates if database is empty
        if (certs.length === 0) {
            const defaults = [
                {
                    title: 'Deep Learning & CV',
                    description: 'Hands-on course focusing on neural network architectures, image processing, convolution operators, and AI-based computer vision.',
                    icon: '👁️',
                    downloadUrl: 'documents/DL_Certificate.pdf'
                },
                {
                    title: 'Machine Learning Program',
                    description: 'Rigorous study covering advanced ML algorithms, feature engineering, model optimization, and deployment pipelines.',
                    icon: '🤖',
                    downloadUrl: 'documents/ML_Certificate.pdf'
                },
                {
                    title: 'Data Science Professional',
                    description: 'Comprehensive training in statistical models, data analysis fundamentals, and real-world ML implementations.',
                    icon: '📊',
                    downloadUrl: 'documents/Nahom_Teshome_Data A.pdf'
                }
            ];
            await Certificate.insertMany(defaults);
            certs = await Certificate.find().sort({ dateAdded: -1 });
        }
        
        res.json({ success: true, data: certs });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// POST /api/certificates - Admin Only (add new certificate)
router.post('/', async (req, res) => {
    if (!isAdmin(req)) {
        return res.status(401).json({ success: false, error: 'Unauthorized' });
    }
    try {
        const { title, description, icon, downloadUrl } = req.body;
        if (!title || !description || !downloadUrl) {
            return res.status(400).json({ success: false, error: 'Missing required fields' });
        }
        const newCert = new Certificate({ title, description, icon, downloadUrl });
        await newCert.save();
        res.status(201).json({ success: true, data: newCert });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// DELETE /api/certificates/:id - Admin Only (delete a certificate)
router.delete('/:id', async (req, res) => {
    if (!isAdmin(req)) {
        return res.status(401).json({ success: false, error: 'Unauthorized' });
    }
    try {
        const cert = await Certificate.findByIdAndDelete(req.params.id);
        if (!cert) {
            return res.status(404).json({ success: false, error: 'Certificate not found' });
        }
        res.json({ success: true });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

module.exports = router;
