// Smooth scroll
function scrollToSection(id) {
    document.getElementById(id).scrollIntoView({ behavior: 'smooth' });
}

// Animate skills when visible
window.addEventListener('scroll', function () {
    const bars = document.querySelectorAll('.progress-bar');
    bars.forEach(bar => {
        const rect = bar.getBoundingClientRect();
        if (rect.top < window.innerHeight) {
            bar.style.width = bar.dataset.width;
        }
    });
});

// Dark mode toggle
document.getElementById("darkModeToggle").addEventListener("click", function () {
    document.body.classList.toggle("dark-mode");
});
// Contact form submission
document.getElementById('contactForm')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const name = document.getElementById('name').value;
    const email = document.getElementById('email').value;
    const message = document.getElementById('message').value;
    const statusDiv = document.getElementById('formStatus');

    try {
        const response = await fetch('http://localhost:5000/api/messages', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, email, message })
        });
        const data = await response.json();
        if (data.success) {
            statusDiv.style.color = 'green';
            statusDiv.textContent = 'Message sent successfully!';
            document.getElementById('contactForm').reset();
        } else {
            statusDiv.style.color = 'red';
            statusDiv.textContent = 'Error sending message.';
        }
    } catch (error) {
        statusDiv.style.color = 'red';
        statusDiv.textContent = 'Network error. Please try again.';
    }
});

// ========== GUESTBOOK / COMMENTS ==========

// Load approved comments on page load
async function loadComments() {
    try {
        const response = await fetch('http://localhost:5000/api/comments');
        const data = await response.json();
        if (data.success) {
            displayComments(data.data);
        }
    } catch (error) {
        console.error('Failed to load comments', error);
    }
}

function displayComments(comments) {
    const container = document.getElementById('commentsList');
    if (!container) return;
    container.innerHTML = '';
    if (comments.length === 0) {
        container.innerHTML = '<p>No comments yet. Be the first!</p>';
        return;
    }
    comments.forEach(cmt => {
        const card = document.createElement('div');
        card.className = 'comment-card';
        card.innerHTML = `
            <p class="comment-name">${escapeHtml(cmt.name)}</p>
            <p>${escapeHtml(cmt.comment)}</p>
            <p class="comment-date">${new Date(cmt.date).toLocaleDateString()}</p>
        `;
        container.appendChild(card);
    });
}

// Simple escape to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Handle comment form submission
document.getElementById('commentForm')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const name = document.getElementById('commentName').value;
    const email = document.getElementById('commentEmail').value;
    const comment = document.getElementById('commentText').value;
    const statusDiv = document.getElementById('commentFormStatus');

    try {
        const response = await fetch('http://localhost:5000/api/comments', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, email, comment })
        });
        const data = await response.json();
        if (data.success) {
            statusDiv.style.color = 'green';
            statusDiv.textContent = 'Comment submitted for approval. Thank you!';
            document.getElementById('commentForm').reset();
        } else {
            statusDiv.style.color = 'red';
            statusDiv.textContent = 'Error submitting comment.';
        }
    } catch (error) {
        statusDiv.style.color = 'red';
        statusDiv.textContent = 'Network error. Please try again.';
    }
});

// Call loadComments when page loads
loadComments();

function displayComments(comments) {
    const container = document.getElementById('commentsList');
    if (!container) return;

    container.innerHTML = '';

    if (comments.length === 0) {
        container.innerHTML = '<p style="text-align: center; color: #888;">No comments yet. Be the first!</p>';
        return;
    }

    comments.forEach(cmt => {
        // Define 'card' inside the loop
        const card = document.createElement('div');
        card.className = 'comment-card';

        // Build the comment HTML (with optional avatar)
        card.innerHTML = `
            <div style="display: flex; gap: 15px; align-items: flex-start;">
                <div style="font-size: 2.5em; color: #00d9ff; min-width: 50px; text-align: center;">👤</div>
                <div style="flex:1;">
                    <p class="comment-name">${escapeHtml(cmt.name)}</p>
                    <p style="margin: 5px 0;">${escapeHtml(cmt.comment)}</p>
                    <p class="comment-date">${new Date(cmt.date).toLocaleDateString()}</p>
                </div>
            </div>
        `;

        container.appendChild(card);
    });
}