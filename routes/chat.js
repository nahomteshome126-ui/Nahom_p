const express = require('express');
const router = express.Router();

const SYSTEM_PROMPT = `
You are the personal AI Assistant of Nahom Teshome. Your goal is to represent Nahom professionally, enthusiastically, and accurately to recruiters, clients, and visitors.

Key Information about Nahom Teshome:
- **Title**: Data Scientist & Full-Stack Software Developer
- **Education**: Data Science Student at Debre Berhan University (DBU), Ethiopia
- **Technical Skills**:
  - Languages & Databases: Python, Power BI, PostgreSQL, C++, R, SQL, JavaScript, TypeScript, MongoDB
  - AI & Data Science: Machine Learning, Deep Learning, TensorFlow, PyTorch, Computer Vision (OpenCV, YOLO), NLP, Pandas, NumPy, Seaborn
  - Web Development: Node.js, Express, React, Vite, HTML5, CSS3, REST APIs
- **Featured Projects**:
  1. *EEP Water Level & Energy Forecasting*: Hydropower forecasting dashboard for Ethiopian Electric Power plants using time-series models (ARIMA/LSTM).
  2. *Computer Vision Detection Model*: Real-time object detection and tracking using CNNs, YOLO, and MobileNet backbones.
  3. *Intelligent Web Applications*: Responsive full-stack web products integrated with MongoDB databases and admin alert systems.
  4. *Reddit & Data Sentiment Analytics*: NLP-driven analytics platform extracting public sentiment trends using Python, Power BI, and PostgreSQL.
- **Contact Details**:
  - Phone: +251 921 971 146
  - WhatsApp: +251 980 291 221 (https://wa.me/251980291221)
  - Email: nahomteshome126@gmail.com
  - Location: Debre Berhan, Amhara, Ethiopia
  - GitHub: https://github.com/nahomteshome126-ui
  - LinkedIn: https://linkedin.com/in/nahomteshome21

Guidelines:
- Keep answers helpful, concise, engaging, and friendly.
- Always answer from Nahom's perspective as his AI assistant.
`;

// Smart Fallback Assistant Knowledge Base when offline or API key pending
function getSmartFallbackResponse(userMessage) {
    const msg = userMessage.toLowerCase();

    if (msg.includes('who') || msg.includes('about') || msg.includes('nahom') || msg.includes('name')) {
        return "Nahom Teshome is a passionate Data Scientist and Full-Stack Software Developer studying Data Science at Debre Berhan University in Ethiopia. He specializes in Machine Learning, Deep Learning, Computer Vision, and modern Web Applications.";
    }

    if (msg.includes('skill') || msg.includes('language') || msg.includes('tool') || msg.includes('tech') || msg.includes('python') || msg.includes('sql') || msg.includes('power bi')) {
        return "Nahom's technical skill set includes Python, Power BI, PostgreSQL, C++, R, SQL, Machine Learning, Deep Learning, PyTorch, TensorFlow, Computer Vision (OpenCV/YOLO), Pandas, JavaScript, React, Node.js, and MongoDB.";
    }

    if (msg.includes('project') || msg.includes('eep') || msg.includes('forecasting') || msg.includes('vision') || msg.includes('app') || msg.includes('sentiment')) {
        return "Nahom has built several featured projects:\n1. EEP Water Level & Energy Forecasting (Hydropower time-series prediction)\n2. Real-Time Computer Vision Detection Model (YOLO/PyTorch object tracking)\n3. Intelligent Full-Stack Web Suite (Node.js, Express, MongoDB)\n4. Reddit & Data Sentiment Analytics (NLP with Python, Power BI & PostgreSQL).";
    }

    if (msg.includes('contact') || msg.includes('phone') || msg.includes('call') || msg.includes('whatsapp') || msg.includes('email') || msg.includes('reach') || msg.includes('location')) {
        return "You can contact Nahom directly via:\n• Phone: +251 921 971 146\n• WhatsApp: +251 980 291 221\n• Email: nahomteshome126@gmail.com\n• Location: Debre Berhan, Ethiopia\nOr fill out the Contact Form on this page!";
    }

    if (msg.includes('resume') || msg.includes('cv') || msg.includes('certificate') || msg.includes('download')) {
        return "You can download Nahom's complete Curriculum Vitae (CV) and professional Data Science, Machine Learning, and Deep Learning certificates directly in the 'Credentials & Resume' section on this page!";
    }

    return "Hello! I am Nahom Teshome's AI Assistant. I can tell you all about his Data Science background, ML/DL projects, technical skills (Python, Power BI, PostgreSQL, C++, R, SQL, etc.), or how to contact him. What would you like to know?";
}

router.post('/', async (req, res) => {
    const { message, history } = req.body;
    if (!message) {
        return res.status(400).json({ success: false, error: 'Message is required' });
    }

    const apiKey = process.env.GEMINI_API_KEY;

    if (!apiKey) {
        // Return smart Knowledge Base response cleanly without demo mode error
        const fallbackMsg = getSmartFallbackResponse(message);
        return res.json({ 
            success: true, 
            message: fallbackMsg 
        });
    }

    try {
        const formattedContents = [];
        if (history && Array.isArray(history)) {
            history.forEach(item => {
                formattedContents.push({
                    role: item.role === 'bot' ? 'model' : 'user',
                    parts: [{ text: item.text }]
                });
            });
        }
        // Add current user message
        formattedContents.push({
            role: 'user',
            parts: [{ text: message }]
        });

        const apiURL = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${apiKey}`;
        
        const response = await fetch(apiURL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                systemInstruction: {
                    parts: [{ text: SYSTEM_PROMPT }]
                },
                contents: formattedContents,
                generationConfig: {
                    maxOutputTokens: 350,
                    temperature: 0.7
                }
            })
        });

        if (!response.ok) {
            const fallbackMsg = getSmartFallbackResponse(message);
            return res.json({ success: true, message: fallbackMsg });
        }

        const data = await response.json();
        
        if (!data.candidates || data.candidates.length === 0) {
            const fallbackMsg = getSmartFallbackResponse(message);
            return res.json({ success: true, message: fallbackMsg });
        }

        const botResponse = data.candidates[0]?.content?.parts?.[0]?.text || getSmartFallbackResponse(message);
        res.json({ success: true, message: botResponse });
    } catch (error) {
        console.error('Chat error fallback:', error);
        const fallbackMsg = getSmartFallbackResponse(message);
        res.json({ success: true, message: fallbackMsg });
    }
});

module.exports = router;
