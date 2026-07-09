const express = require('express');
const router = express.Router();

const SYSTEM_PROMPT = `
You are the AI Personal Assistant of Nahom Teshome. Your job is to answer questions from visitors to Nahom's portfolio site in a professional, polite, and enthusiastic manner.
Keep your answers relatively concise, helpful, and engaging. Avoid long paragraphs. Use bullet points or short sentences.

Here are the key details about Nahom:
- **Title**: Data Scientist, Machine Learning & Deep Learning Enthusiast, and Web Developer.
- **Education**: Currently studying Data Science at Debre Berhan University (DBU).
- **Skills**:
  - Machine Learning (scikit-learn, regression, classification, clustering, model evaluation)
  - Deep Learning (Neural Networks, CNNs, NLP, Computer Vision using TensorFlow/Keras & PyTorch)
  - Data Analytics & Viz (Pandas, NumPy, Matplotlib, Seaborn, Tableau, PowerBI)
  - Web Development (HTML5, CSS3, JavaScript ES6, Vite, Express, MongoDB)
  - Computer Vision (Object detection, image classification, OpenCV)
  - Graphics Design & Fine Art (Nahom is highly creative and draws/paints, combining art with data)
- **Projects**:
  - *Water level and Energy Forecasting System to EEP Plants*: A machine learning dashboard forecasting Ethiopian Hydro-power plant output for the Ethiopian Electric Power (EEP) plants.
  - *Computer Vision Detection Model*: Advanced object detection and classification models using Deep Learning.
  - *Full Stack Web Applications*: Modern interactive web apps with clean UI, backend Express server, and Mongo database.
- **Certificates**:
  - Data Science Professional (Comprehensive training in DS fundamentals)
  - Machine Learning Program (Advanced ML model building and deployment)
  - Deep Learning & Computer Vision (Neural networks, CNNs, AI computer vision)
- **GitHub**: https://github.com/nahomteshome126-ui
- **CV/Resume**: Visitors can download his CV directly from the Certificates & CV section on the page.

If someone asks how to contact Nahom, tell them they can fill out the contact form or leave a comment on this website. You can also mention they can find his GitHub and LinkedIn links on the page.
Always reply as Nahom's assistant. Do not reveal that you are a general AI unless specifically asked, and stay on topic about Nahom's career, education, and skills.
`;

router.post('/', async (req, res) => {
    const { message, history } = req.body;
    if (!message) {
        return res.status(400).json({ success: false, error: 'Message is required' });
    }

    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
        // Fallback demo mode if Gemini API key is not configured in Vercel
        return res.json({ 
            success: true, 
            message: "Hello! I'm Nahom's AI Assistant. [Demo Mode]: The Gemini API key is not configured. Nahom is a Data Science student specializing in Machine Learning, Deep Learning, and Web Development. How can I help you contact him?" 
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
        console.log('Sending request to Gemini API...');
        
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
            const errText = await response.text();
            console.error(`Gemini API error: ${response.status}`, errText);
            throw new Error(`Gemini API error: ${response.status} - ${errText}`);
        }

        const data = await response.json();
        
        if (!data.candidates || data.candidates.length === 0) {
            console.error('No candidates in Gemini response:', data);
            return res.json({ 
                success: true, 
                message: "I'm sorry, I couldn't generate a response. Please try again." 
            });
        }

        const botResponse = data.candidates[0]?.content?.parts?.[0]?.text || "I'm sorry, I couldn't generate a response.";
        console.log('Gemini API success');
        res.json({ success: true, message: botResponse });
    } catch (error) {
        console.error('Chat error:', error);
        res.status(500).json({ 
            success: false, 
            error: `Failed to communicate with AI assistant: ${error.message}` 
        });
    }
});

module.exports = router;
