// ================= NAVBAR SCROLL EFFECT =================

window.addEventListener('scroll', () => {
    const navbar = document.querySelector('.navbar');
    if (window.scrollY > 50) {
        navbar.classList.add('scrolled');
    } else {
        navbar.classList.remove('scrolled');
    }
});

// ================= MOBILE NAVIGATION MENU =================

const mobileMenuBtn = document.querySelector('.mobile-menu-btn');
const navLinks = document.querySelector('.nav-links');

if (mobileMenuBtn && navLinks) {
    mobileMenuBtn.addEventListener('click', () => {
        navLinks.classList.toggle('active');
        mobileMenuBtn.classList.toggle('open');
        
        // Animating menu bars
        const bars = mobileMenuBtn.querySelectorAll('.bar');
        if (mobileMenuBtn.classList.contains('open')) {
            bars[0].style.transform = 'rotate(-45deg) translate(-5px, 6px)';
            bars[1].style.opacity = '0';
            bars[2].style.transform = 'rotate(45deg) translate(-5px, -6px)';
        } else {
            bars[0].style.transform = 'none';
            bars[1].style.opacity = '1';
            bars[2].style.transform = 'none';
        }
    });

    // Close menu when link is clicked
    navLinks.querySelectorAll('a').forEach(link => {
        link.addEventListener('click', () => {
            navLinks.classList.remove('active');
            mobileMenuBtn.classList.remove('open');
            mobileMenuBtn.querySelectorAll('.bar').forEach(bar => bar.style.transform = 'none');
            mobileMenuBtn.querySelectorAll('.bar')[1].style.opacity = '1';
        });
    });
}

// ================= HERO DYNAMIC NEURAL CANVAS =================

const canvas = document.getElementById('hero-canvas');
if (canvas) {
    const ctx = canvas.getContext('2d');
    let particles = [];
    const maxParticles = 60;
    const connectionDist = 120;

    function resizeCanvas() {
        canvas.width = canvas.parentElement.offsetWidth;
        canvas.height = canvas.parentElement.offsetHeight;
    }
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    class Particle {
        constructor() {
            this.x = Math.random() * canvas.width;
            this.y = Math.random() * canvas.height;
            this.vx = (Math.random() - 0.5) * 0.6;
            this.vy = (Math.random() - 0.5) * 0.6;
            this.radius = Math.random() * 2 + 1.5;
        }

        update() {
            this.x += this.vx;
            this.y += this.vy;

            // Bounce on boundaries
            if (this.x < 0 || this.x > canvas.width) this.vx *= -1;
            if (this.y < 0 || this.y > canvas.height) this.vy *= -1;
        }

        draw() {
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
            const isDark = document.body.classList.contains('dark-mode');
            ctx.fillStyle = isDark ? 'rgba(56, 189, 248, 0.6)' : 'rgba(14, 165, 233, 0.4)';
            ctx.fill();
        }
    }

    // Populate particles
    for (let i = 0; i < maxParticles; i++) {
        particles.push(new Particle());
    }

    function animateCanvas() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const isDark = document.body.classList.contains('dark-mode');
        
        // Draw and update particles
        particles.forEach(p => {
            p.update();
            p.draw();
        });

        // Draw connections
        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy);

                if (dist < connectionDist) {
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    const alpha = (1 - dist / connectionDist) * 0.15;
                    ctx.strokeStyle = isDark 
                        ? `rgba(56, 189, 248, ${alpha})` 
                        : `rgba(14, 165, 233, ${alpha})`;
                    ctx.lineWidth = 1;
                    ctx.stroke();
                }
            }
        }
        requestAnimationFrame(animateCanvas);
    }
    animateCanvas();
}

// ================= SCROLL AND SMOOTH SCROLL UTILS =================

function scrollToSection(id) {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' });
}

// Animate skill bars when scrolled into view
const observeSkills = () => {
    const bars = document.querySelectorAll('.progress-bar');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const bar = entry.target;
                bar.style.width = bar.dataset.width;
                observer.unobserve(bar);
            }
        });
    }, { threshold: 0.1 });

    bars.forEach(bar => observer.observe(bar));
};
observeSkills();

// ================= THEME TOGGLE (DARK/LIGHT MODE) =================

const darkModeToggle = document.getElementById("darkModeToggle");
// Load saved preference
if (localStorage.getItem('theme') === 'dark' || 
    (!localStorage.getItem('theme') && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
    document.body.classList.add('dark-mode');
    if (darkModeToggle) darkModeToggle.textContent = "☀️";
} else {
    document.body.classList.remove('dark-mode');
    if (darkModeToggle) darkModeToggle.textContent = "🌙";
}

if (darkModeToggle) {
    darkModeToggle.addEventListener("click", () => {
        document.body.classList.toggle("dark-mode");
        const isDark = document.body.classList.contains("dark-mode");
        darkModeToggle.textContent = isDark ? "☀️" : "🌙";
        localStorage.setItem('theme', isDark ? 'dark' : 'light');
    });
}

// ================= CONTACT FORM =================

const contactForm = document.getElementById('contactForm');
if (contactForm) {
    contactForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const name = document.getElementById('name').value.trim();
        const email = document.getElementById('email').value.trim();
        const message = document.getElementById('message').value.trim();
        const statusDiv = document.getElementById('formStatus');
        const submitBtn = contactForm.querySelector('.btn-submit');

        if (!name || !email || !message) return;

        // Set Loading state
        submitBtn.disabled = true;
        const origText = submitBtn.textContent;
        submitBtn.textContent = 'Sending Message...';
        statusDiv.textContent = '';

        try {
            const response = await fetch('/api/messages', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, email, message })
            });
            const data = await response.json();
            if (data.success) {
                statusDiv.style.color = '#10b981'; // Green
                statusDiv.textContent = '✅ Message sent successfully! Nahom will review it soon.';
                contactForm.reset();
            } else {
                statusDiv.style.color = '#ef4444'; // Red
                statusDiv.textContent = `❌ Error: ${data.error || 'Failed to send message.'}`;
            }
        } catch (error) {
            statusDiv.style.color = '#ef4444';
            statusDiv.textContent = '❌ Connection error. Please verify the server is running.';
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = origText;
        }
    });
}

// ================= GUESTBOOK (COMMENTS) =================

const commentForm = document.getElementById('commentForm');

async function loadComments() {
    const container = document.getElementById('commentsList');
    if (!container) return;

    container.innerHTML = '<p style="text-align: center; color: var(--text-muted);">Loading comments...</p>';

    try {
        const response = await fetch('/api/comments');
        const data = await response.json();
        if (data.success) {
            displayComments(data.data);
        } else {
            container.innerHTML = '<p style="text-align: center; color: #ef4444;">Failed to load comments.</p>';
        }
    } catch (error) {
        console.error('Failed to load comments', error);
        container.innerHTML = '<p style="text-align: center; color: var(--text-muted);">Guestbook database offline. Be the first to start it!</p>';
    }
}

function displayComments(comments) {
    const container = document.getElementById('commentsList');
    if (!container) return;

    container.innerHTML = '';

    if (!comments || comments.length === 0) {
        container.innerHTML = '<p style="text-align: center; color: var(--text-muted);">No approved comments yet. Be the first!</p>';
        return;
    }

    comments.forEach(cmt => {
        const card = document.createElement('div');
        card.className = 'comment-card';

        // Escaping HTML to prevent XSS
        const name = escapeHtml(cmt.name);
        const comment = escapeHtml(cmt.comment);
        const dateStr = new Date(cmt.date).toLocaleDateString(undefined, { 
            year: 'numeric', 
            month: 'short', 
            day: 'numeric' 
        });

        card.innerHTML = `
            <div style="display: flex; gap: 15px; align-items: flex-start;">
                <div style="font-size: 1.8rem; background: var(--bg-alt); width: 45px; height: 45px; border-radius: 50%; display: flex; align-items: center; justify-content: center; border: 1px solid var(--glass-border);">👤</div>
                <div style="flex: 1;">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                        <span class="comment-name">${name}</span>
                        <span class="comment-date">${dateStr}</span>
                    </div>
                    <p class="comment-text" style="margin-top: 8px;">${comment}</p>
                </div>
            </div>
        `;
        container.appendChild(card);
    });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

if (commentForm) {
    commentForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const name = document.getElementById('commentName').value.trim();
        const email = document.getElementById('commentEmail').value.trim();
        const comment = document.getElementById('commentText').value.trim();
        const statusDiv = document.getElementById('commentFormStatus');
        const submitBtn = commentForm.querySelector('.btn-submit');

        if (!name || !email || !comment) return;

        submitBtn.disabled = true;
        const origText = submitBtn.textContent;
        submitBtn.textContent = 'Posting...';
        statusDiv.textContent = '';

        try {
            const response = await fetch('/api/comments', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, email, comment })
            });
            const data = await response.json();
            if (data.success) {
                statusDiv.style.color = '#10b981';
                statusDiv.textContent = '✅ Comment submitted! It will appear once approved by Nahom.';
                commentForm.reset();
            } else {
                statusDiv.style.color = '#ef4444';
                statusDiv.textContent = `❌ Error: ${data.error || 'Failed to submit comment.'}`;
            }
        } catch (error) {
            statusDiv.style.color = '#ef4444';
            statusDiv.textContent = '❌ Connection error. Failed to post comment.';
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = origText;
        }
    });
}

// Initial load
loadComments();

// ================= AI CHATBOT CONTROLLER =================

const chatToggle = document.getElementById('chat-toggle');
const chatClose = document.getElementById('chat-close');
const chatWindow = document.getElementById('chat-window');
const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chat-input');
const chatMessages = document.getElementById('chat-messages');
const chatSuggestions = document.getElementById('chat-suggestions');

let chatHistory = []; // Tracks [{role: 'user'|'bot', text: '...'}]

if (chatToggle && chatClose && chatWindow) {
    // Open/Close chat window
    chatToggle.addEventListener('click', () => {
        chatWindow.classList.toggle('active');
        // Hide badge pulse when user interacts
        const pulse = chatToggle.querySelector('.chat-pulse');
        if (pulse) pulse.style.display = 'none';
        
        if (chatWindow.classList.contains('active')) {
            chatInput.focus();
        }
    });

    chatClose.addEventListener('click', () => {
        chatWindow.classList.remove('active');
    });

    // Handle suggestion chips
    document.querySelectorAll('.chat-suggest-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const text = btn.textContent;
            sendUserMessage(text);
        });
    });

    // Handle form submit
    chatForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const text = chatInput.value.trim();
        if (!text) return;
        sendUserMessage(text);
        chatInput.value = '';
    });
}

function appendMessage(role, text) {
    if (!chatMessages) return;
    const bubble = document.createElement('div');
    bubble.className = `chat-bubble ${role}`;
    
    // Convert newlines to breaks or format markdown links
    let formattedText = escapeHtml(text).replace(/\n/g, '<br>');
    // Simple regex to parse markdown links like [Text](URL)
    formattedText = formattedText.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer" style="color: inherit; text-decoration: underline; font-weight: 600;">$1</a>');
    
    bubble.innerHTML = `<p>${formattedText}</p>`;
    chatMessages.appendChild(bubble);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function appendLoading() {
    if (!chatMessages) return null;
    const bubble = document.createElement('div');
    bubble.className = 'chat-bubble bot loading';
    bubble.id = 'chat-loader';
    bubble.innerHTML = '<span></span><span></span><span></span>';
    chatMessages.appendChild(bubble);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return bubble;
}

async function sendUserMessage(text) {
    appendMessage('user', text);
    chatHistory.push({ role: 'user', text });

    const loader = appendLoading();

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text, history: chatHistory })
        });
        const data = await response.json();
        
        if (loader) loader.remove();

        if (data.success) {
            appendMessage('bot', data.message);
            chatHistory.push({ role: 'bot', text: data.message });
        } else {
            // Show the actual error message from the backend
            const errorMsg = data.error || 'An error occurred while communicating with the AI assistant.';
            appendMessage('bot', `⚠️ Error: ${errorMsg}`);
            console.error('API Error:', data.error);
        }
    } catch (error) {
        if (loader) loader.remove();
        console.error('Chatbot API error:', error);
        appendMessage('bot', `Connection failed: ${error.message}`);
    }
}

// ================= PROJECT MODAL CONTROLLER =================

const projectModal = document.getElementById('projectModal');
const modalBody = document.getElementById('modalBody');

const PROJECT_DATA = {
    eep: {
        title: "Water Reservoir & Energy Forecasting System",
        tag: "Machine Learning & Forecasting",
        desc: "A decision support forecasting tool built for the Ethiopian Electric Power (EEP) plants. The system ingests geographical water levels, rainfall, and historic power generation files to project upcoming metrics.",
        features: [
            "Predicts daily/weekly water reservoir inflow speeds.",
            "Forecasts energy output utilizing deep Recurrent Neural Networks (LSTM).",
            "Responsive web dashboard built with Flask and chart visualizations.",
            "Assists grid dispatch planners in preventing overflow releases and optimizing generator output."
        ]
    },
    cv: {
        title: "Computer Vision Detection Model",
        tag: "Deep Learning & Neural Networks",
        desc: "An intelligent, real-time object tracking application designed using state-of-the-art Convolutional Neural Networks.",
        features: [
            "Fine-tuned YOLO (You Only Look Once) framework for high-precision bounding boxes.",
            "Runs edge-optimized detection speeds utilizing MobileNet backbones.",
            "Processes live camera video streams and extracts visual tag indexes.",
            "Integrated image segmentation filters using PyTorch modules."
        ]
    },
    web: {
        title: "Intelligent Web Applications",
        tag: "Full-Stack Software Engineering",
        desc: "A suite of responsive, full-stack software products built around secure MongoDB databases and clean user dashboards.",
        features: [
            "Engineered robust REST API servers using Node.js and Express.",
            "Created modular design files using modern front-end build tools like Vite.",
            "Implemented admin dashboard security layers including password verification.",
            "Clean visual animations, dark modes, and complete layout responsiveness."
        ]
    }
};

function openProjectModal(key) {
    if (!projectModal || !modalBody || !PROJECT_DATA[key]) return;
    
    const data = PROJECT_DATA[key];
    
    let featuresHTML = '';
    data.features.forEach(feat => {
        featuresHTML += `<li>${escapeHtml(feat)}</li>`;
    });

    modalBody.innerHTML = `
        <span class="modal-tag">${escapeHtml(data.tag)}</span>
        <h3 class="modal-title">${escapeHtml(data.title)}</h3>
        <p class="modal-desc">${escapeHtml(data.desc)}</p>
        <div class="modal-features">
            <h4>Key Capabilities</h4>
            <ul>
                ${featuresHTML}
            </ul>
        </div>
    `;

    projectModal.classList.add('active');
    document.body.style.overflow = 'hidden'; // Lock background scrolling
}

function closeProjectModal() {
    if (!projectModal) return;
    projectModal.classList.remove('active');
    document.body.style.overflow = ''; // Unlock background scrolling
}

// Close modal when clicking outside contents
if (projectModal) {
    projectModal.addEventListener('click', (e) => {
        if (e.target === projectModal) {
            closeProjectModal();
        }
    });
}