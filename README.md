# My portifolio

## Overview

My portifolio is a personal portfolio website for Nahom Teshome. It presents professional information, featured projects, technical skills, certificates, a downloadable CV, a contact form, and a guestbook/comments section. The project also includes a simple admin dashboard for managing contact messages and visitor comments.

## Project Goals

- Introduce the portfolio owner with a clear landing page
- Showcase skills, projects, certificates, and CV
- Allow visitors to send contact messages
- Allow visitors to leave comments for approval
- Provide an admin page to manage messages and comments

## Technologies Used

- HTML5
- CSS3
- JavaScript (Vanilla JS)

## Main Features

- Responsive single-page portfolio layout
- Sticky navigation bar
- Hero section with profile image and social links
- About section
- Animated skill progress bars
- Projects showcase cards
- Certificates and CV download section
- Dark mode toggle
- Contact form connected to a backend API
- Guestbook/comment system connected to a backend API
- Admin dashboard for:
  - Viewing messages
  - Archiving and unarchiving messages
  - Viewing comments
  - Approving and unapproving comments
  - Deleting comments

## Project Structure

```text
Nahom_P/
|-- index.html
|-- admin.html
|-- style.css
|-- script.js
|-- images/
|-- documents/
```

## File Description

### `index.html`

This is the main portfolio page. It contains:

- Navigation bar
- Hero section
- About section
- Skills section
- Projects section
- Certificates and CV section
- Guestbook/comments section
- Contact section
- Footer

### `admin.html`

This is the admin dashboard page. It allows the admin to:

- Log in using a password
- View all contact messages
- Archive or unarchive messages
- View all comments
- Approve, unapprove, or delete comments

### `style.css`

This file controls the visual design of the site, including:

- Layout and spacing
- Hero section styling
- Buttons and cards
- Guestbook styling
- Contact form styling
- Dark mode styling
- Responsive behavior for smaller screens

### `script.js`

This file handles interactivity, including:

- Smooth scrolling to sections
- Skill bar animation on scroll
- Dark mode toggle
- Contact form submission
- Loading approved comments
- Posting new comments
- Escaping comment content for basic XSS protection

## Backend/API Requirements

This frontend expects a backend server running on:

```text
http://localhost:5000
```

The following API endpoints are used:

### Contact messages

- `POST /api/messages` - submit a message from the contact form
- `GET /api/messages?password=...&includeArchived=true` - load messages in admin view
- `PATCH /api/messages/:id/archive?password=...` - archive or unarchive a message

### Comments

- `GET /api/comments` - load approved comments on the public page
- `POST /api/comments` - submit a new comment
- `GET /api/comments/all?password=...` - load all comments in admin view
- `PATCH /api/comments/:id/approve?password=...` - approve or unapprove a comment
- `DELETE /api/comments/:id?password=...` - delete a comment

## How to Run the Project

1. Place the project files in a local folder.
2. Make sure the `images` and `documents` folders contain the required assets.
3. Open `index.html` in a browser to view the portfolio.
4. Open `admin.html` in a browser to use the admin dashboard.
5. Start the backend server on `http://localhost:5000` if you want the contact form and comments to work.

## Assets Used

### Images

- `images/Nahom.png` for the profile picture

### Documents

- `documents/CV.pdf` for CV download
- Certificate PDF files for the certificates section

## Notes

- Without the backend server, the contact form and comments system will not work.
- The dark mode feature is handled by toggling the `dark-mode` class on the `body`.
- The comments section only displays approved comments from the backend.
- The admin dashboard is protected only by a password passed to the backend API, so security depends on backend validation.

## Known Improvements

- Replace placeholder links such as the LinkedIn profile URL
- Improve encoding issues showing some icons and special characters
- Add better admin authentication
- Add form validation feedback before submission
- Add project detail links or pages for each project card
- Add deployment instructions

## Author

Nahom Teshome
