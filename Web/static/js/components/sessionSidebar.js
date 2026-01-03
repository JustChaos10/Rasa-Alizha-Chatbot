/**
 * Session Sidebar Component
 * 
 * Manages chat history sidebar with multiple sessions per user.
 * ChatGPT-like conversation history panel.
 */

// Session state
let sessions = [];
let activeSessionId = null;
let sidebarOpen = false;

/**
 * Initialize the session sidebar
 */
async function initSessionSidebar() {
    console.log('🔧 Initializing session sidebar...');
    
    const toggleBtn = document.getElementById('toggleSidebar');
    const closeBtn = document.getElementById('closeSidebar');
    const newChatBtn = document.getElementById('newChatBtn');
    const sidebar = document.getElementById('sessionsSidebar');
    
    console.log('Toggle button found:', !!toggleBtn);
    console.log('Sidebar found:', !!sidebar);
    
    if (!toggleBtn || !sidebar) {
        console.error('❌ Session sidebar elements not found! Check HTML structure.');
        return;
    }
    
    // Toggle sidebar
    toggleBtn.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        console.log('🔘 Toggle button clicked');
        toggleSidebar();
    });
    
    // Close sidebar
    if (closeBtn) {
        closeBtn.addEventListener('click', (e) => {
            e.preventDefault();
            console.log('❌ Close button clicked');
            closeSidebar();
        });
    }
    
    // New chat button
    if (newChatBtn) {
        newChatBtn.addEventListener('click', async (e) => {
            e.preventDefault();
            console.log('➕ New chat button clicked');
            await createNewSession();
        });
    }
    
    // Load sessions on init
    await loadSessions();
    
    // Initialize resize functionality
    initSidebarResize();
    
    console.log('✅ Session sidebar initialized successfully');
}

/**
 * Initialize sidebar resize functionality
 */
function initSidebarResize() {
    const sidebar = document.getElementById('sessionsSidebar');
    if (!sidebar) return;

    // Create resizer handle if it doesn't exist
    let resizer = sidebar.querySelector('.sidebar-resizer');
    if (!resizer) {
        resizer = document.createElement('div');
        resizer.className = 'sidebar-resizer';
        sidebar.appendChild(resizer);
    }

    let isResizing = false;

    resizer.addEventListener('mousedown', (e) => {
        isResizing = true;
        document.addEventListener('mousemove', resize);
        document.addEventListener('mouseup', stopResize);
        document.body.style.cursor = 'col-resize';
        e.preventDefault(); // Prevent text selection
    });

    function resize(e) {
        if (!isResizing) return;
        // Calculate new width based on mouse position
        const newWidth = e.clientX; 
        // Set limits (min 200px, max 600px)
        if (newWidth > 200 && newWidth < 600) {
            sidebar.style.width = `${newWidth}px`;
        }
    }

    function stopResize() {
        isResizing = false;
        document.removeEventListener('mousemove', resize);
        document.removeEventListener('mouseup', stopResize);
        document.body.style.cursor = 'default';
    }
}

/**
 * Toggle sidebar visibility
 */
function toggleSidebar() {
    console.log('🔄 toggleSidebar called, current state:', sidebarOpen);
    
    if (sidebarOpen) {
        closeSidebar();
    } else {
        openSidebar();
    }
}

/**
 * Open sidebar
 */
function openSidebar() {
    console.log('📂 Opening sidebar...');
    const sidebar = document.getElementById('sessionsSidebar');
    const toggleBtn = document.getElementById('toggleSidebar');
    
    if (!sidebar) {
        console.error('❌ Sidebar element not found!');
        return;
    }
    
    sidebar.classList.add('open');
    
    if (toggleBtn) {
        toggleBtn.classList.add('active');
    }
    sidebarOpen = true;
    
    console.log('✅ Sidebar opened');
    
    // Refresh sessions when opening
    loadSessions();
}

/**
 * Close sidebar
 */
function closeSidebar() {
    console.log('📁 Closing sidebar...');
    const sidebar = document.getElementById('sessionsSidebar');
    const toggleBtn = document.getElementById('toggleSidebar');
    
    if (sidebar) {
        sidebar.classList.remove('open');
    }
    if (toggleBtn) {
        toggleBtn.classList.remove('active');
    }
    sidebarOpen = false;
}

/**
 * Load all sessions from API
 */
async function loadSessions() {
    console.log('📡 Loading sessions from API...');
    const sessionsList = document.getElementById('sessionsList');
    
    try {
        const response = await fetch('/api/sessions');
        console.log('📡 API response status:', response.status);
        const data = await response.json();
        console.log('📡 API response data:', data);
        
        if (data.success) {
            sessions = data.sessions || [];
            activeSessionId = data.active_session_id;
            console.log(`📋 Loaded ${sessions.length} sessions, active: ${activeSessionId}`);
            renderSessions();
        } else {
            console.error('❌ Failed to load sessions:', data.error);
            showSessionError('Failed to load chat history');
        }
    } catch (error) {
        console.error('❌ Error loading sessions:', error);
        showSessionError('Error loading chat history');
    }
}

/**
 * Render sessions list
 */
function renderSessions() {
    const sessionsList = document.getElementById('sessionsList');
    
    if (!sessionsList) return;
    
    console.log('🎨 Rendering sessions, count:', sessions.length);
    
    if (sessions.length === 0) {
        sessionsList.innerHTML = `
            <div class="sessions-empty">
                <i class="material-icons">chat_bubble_outline</i>
                <p>No chat history yet</p>
                <p style="color: #999; font-size: 14px;">Start a new conversation!</p>
            </div>
        `;
        return;
    }
    
    let html = '';
    sessions.forEach(session => {
        const isActive = session.session_id === activeSessionId;
        const date = formatSessionDate(session.last_activity);
        const preview = session.preview || 'New conversation';
        // Use session name, but if it's "New Chat" and there's a preview, use the preview
        const displayName = (session.name === 'New Chat' && session.preview) 
            ? (session.preview.length > 30 ? session.preview.substring(0, 30) + '...' : session.preview)
            : session.name;
        
        html += `
            <div class="session-item ${isActive ? 'active' : ''}" 
                 data-session-id="${session.session_id}"
                 onclick="switchToSession('${session.session_id}')"
                 style="cursor: pointer;">
                <div class="session-content">
                    <div class="session-name">${escapeHtml(displayName)}</div>
                    <div class="session-meta">
                        <span class="session-date">${date}</span>
                    </div>
                </div>
                <div class="session-actions-btns">
                    <button class="session-action-btn" onclick="event.stopPropagation(); showRenameModal('${session.session_id}', '${escapeHtml(session.name).replace(/'/g, "\\'")}')" title="Rename">
                        <i class="material-icons tiny">edit</i>
                    </button>
                    <button class="session-action-btn delete" onclick="event.stopPropagation(); confirmDeleteSession('${session.session_id}')" title="Delete">
                        <i class="material-icons tiny">delete</i>
                    </button>
                </div>
            </div>
        `;
    });
    
    sessionsList.innerHTML = html;
}

/**
 * Format session date for display
 */
function formatSessionDate(timestamp) {
    if (!timestamp) return '';
    
    const date = new Date(timestamp * 1000);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);
    
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Create a new chat session
 */
async function createNewSession() {
    try {
        const response = await fetch('/api/sessions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        });
        
        const data = await response.json();
        
        if (data.success) {
            activeSessionId = data.session.session_id;
            
            // Clear the chat UI
            clearChatUI();
            
            // Show greeting
            showGreeting();
            
            // Reload sessions list
            await loadSessions();
            
            // Toast notification
            if (window.M) {
                M.toast({ html: '✨ New chat started', classes: 'green' });
            }
        } else {
            console.error('Failed to create session:', data.error);
            if (window.M) {
                M.toast({ html: 'Failed to create new chat', classes: 'red' });
            }
        }
    } catch (error) {
        console.error('Error creating session:', error);
        if (window.M) {
            M.toast({ html: 'Error creating new chat', classes: 'red' });
        }
    }
}

/**
 * Switch to a different session
 */
async function switchToSession(sessionId) {
    console.log('🔄 switchToSession called with:', sessionId);
    console.log('📍 Current activeSessionId:', activeSessionId);
    
    if (sessionId === activeSessionId) {
        console.log('⏭️ Already on this session, skipping');
        return;
    }
    
    try {
        console.log('📡 Fetching session data...');
        const response = await fetch(`/api/sessions/${sessionId}/switch`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        
        const data = await response.json();
        console.log('📥 Switch response:', data);
        
        if (data.success) {
            activeSessionId = sessionId;
            
            // Clear current chat and load session messages
            console.log('🧹 Clearing chat UI...');
            clearChatUI();
            
            // Render session messages
            if (data.messages && data.messages.length > 0) {
                console.log(`📜 Rendering ${data.messages.length} messages`);
                renderSessionMessages(data.messages);
            } else {
                console.log('👋 No messages, showing greeting');
                showGreeting();
            }
            
            // Update active state in sidebar
            updateActiveSession(sessionId);
            
            // Close sidebar on mobile
            if (window.innerWidth <= 768) {
                closeSidebar();
            }
            
            console.log('✅ Session switch complete');
        } else {
            console.error('❌ Failed to switch session:', data.error);
            if (window.M) {
                M.toast({ html: 'Failed to switch chat', classes: 'red' });
            }
        }
    } catch (error) {
        console.error('❌ Error switching session:', error);
        if (window.M) {
            M.toast({ html: 'Error switching chat', classes: 'red' });
        }
    }
}

/**
 * Render messages from a session
 */
function renderSessionMessages(messages) {
    const chats = document.getElementById('chats');
    if (!chats) return;
    
    messages.forEach(msg => {
        if (msg.role === 'user') {
            // Render user message with proper avatar (matching the live chat format)
            const userHtml = `
                <img class="userAvatar" src="/static/images/userAvatar.jpg" />
                <p class="userMsg">${escapeHtml(msg.content)}</p>
                <div class="clearfix"></div>
            `;
            chats.insertAdjacentHTML('beforeend', userHtml);
        } else if (msg.role === 'assistant') {
            // Check if this is an adaptive card message
            const metadata = msg.metadata || {};
            
            if (metadata.type === 'adaptive_card' && metadata.card) {
                // Render the adaptive card using the stored payload
                const payload = {
                    data: {
                        type: 'adaptive_card',
                        card: metadata.card,
                        metadata: metadata.card_metadata || {},
                        message: msg.content
                    }
                };
                
                if (window.renderAdaptiveCardPayload) {
                    window.renderAdaptiveCardPayload(payload);
                } else {
                    // Fallback to text if renderer not available
                    const botHtml = `
                        <img class="botAvatar" src="/static/images/aliza-icon.jpg" />
                        <span class="botMsg">${formatBotMessage(msg.content)}</span>
                        <div class="clearfix"></div>
                    `;
                    chats.insertAdjacentHTML('beforeend', botHtml);
                }
            } else {
                // Regular text message - render the bot message first
                const botHtml = `
                    <img class="botAvatar" src="/static/images/aliza-icon.jpg" />
                    <div class="botMsg">${formatBotMessage(msg.content)}</div>
                    <div class="clearfix"></div>
                `;
                chats.insertAdjacentHTML('beforeend', botHtml);
                
                // Check for image_data in metadata and render as separate visual panel
                // (matching the structure from visualPanel.js)
                if (metadata.image_data) {
                    const img = metadata.image_data;
                    const imageUrl = img.image_url || img.thumbnail_url || img.url || '';
                    const title = escapeHtml(img.title || '');
                    const alt = escapeHtml(img.alt || title || 'Image');
                    const summary = escapeHtml(img.summary || '');
                    const attribution = escapeHtml(img.attribution || '');
                    const sourceUrl = img.source_url ? escapeHtml(img.source_url) : '';
                    
                    // Build caption sections
                    let captionHtml = '';
                    if (summary) {
                        captionHtml += `<p class="visual-panel__summary">${summary}</p>`;
                    }
                    if (attribution || sourceUrl) {
                        const sourceLabel = attribution || 'Source';
                        const sourceMarkup = sourceUrl
                            ? `<a href="${sourceUrl}" target="_blank" rel="noopener noreferrer">${sourceLabel}</a>`
                            : sourceLabel;
                        captionHtml += `<p class="visual-panel__source">${sourceMarkup}</p>`;
                    }
                    
                    const visualPanelHtml = `
                        <div class="visual-panel">
                            <div class="visual-panel__image-wrapper">
                                <img src="${imageUrl}" alt="${alt}" class="visual-panel__image" loading="lazy">
                            </div>
                            ${title ? `<h4 class="visual-panel__title">${title}</h4>` : ''}
                            ${captionHtml}
                        </div>
                    `;
                    chats.insertAdjacentHTML('beforeend', visualPanelHtml);
                }
                
                // Check for chart_path in metadata and render inline chart
                if (metadata.chart_path) {
                    let chartUrl = metadata.chart_path;
                    // Convert file path to URL if needed
                    if (chartUrl.startsWith('charts/')) {
                        chartUrl = '/' + chartUrl;
                    } else if (!chartUrl.startsWith('/')) {
                        chartUrl = '/charts/' + chartUrl;
                    }
                    
                    const inlineChartHtml = `
                        <div class="inline-chart">
                            <img src="${chartUrl}" alt="Data visualization chart" class="inline-chart__image" loading="lazy">
                        </div>
                    `;
                    chats.insertAdjacentHTML('beforeend', inlineChartHtml);
                }
            }
        }
    });
    
    scrollToBottomOfResults();
}

/**
 * Format bot message (basic markdown support)
 */
function formatBotMessage(content) {
    if (!content) return '';
    
    // Use showdown if available, otherwise escape HTML
    if (window.showdown) {
        const converter = new showdown.Converter();
        return converter.makeHtml(content);
    }
    
    // Fallback: preserve newlines if showdown not available
    return escapeHtml(content).replace(/\n/g, '<br>');
}

/**
 * Update active session in sidebar
 */
function updateActiveSession(sessionId) {
    const items = document.querySelectorAll('.session-item');
    items.forEach(item => {
        if (item.dataset.sessionId === sessionId) {
            item.classList.add('active');
        } else {
            item.classList.remove('active');
        }
    });
}

/**
 * Show rename modal
 */
function showRenameModal(sessionId, currentName) {
    const newName = prompt('Rename chat:', currentName);
    if (newName && newName.trim() !== currentName) {
        renameSession(sessionId, newName.trim());
    }
}

/**
 * Rename a session
 */
async function renameSession(sessionId, newName) {
    try {
        const response = await fetch(`/api/sessions/${sessionId}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: newName })
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Reload sessions
            await loadSessions();
            
            if (window.M) {
                M.toast({ html: 'Chat renamed', classes: 'green' });
            }
        } else {
            console.error('Failed to rename session:', data.error);
            if (window.M) {
                M.toast({ html: 'Failed to rename chat', classes: 'red' });
            }
        }
    } catch (error) {
        console.error('Error renaming session:', error);
        if (window.M) {
            M.toast({ html: 'Error renaming chat', classes: 'red' });
        }
    }
}

/**
 * Confirm and delete a session
 */
function confirmDeleteSession(sessionId) {
    if (confirm('Are you sure you want to delete this chat? This cannot be undone.')) {
        deleteSession(sessionId);
    }
}

/**
 * Delete a session
 */
async function deleteSession(sessionId) {
    try {
        const response = await fetch(`/api/sessions/${sessionId}`, {
            method: 'DELETE'
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Reload sessions
            await loadSessions();
            
            // If we deleted the active session, get the new active one
            if (sessionId === activeSessionId) {
                await loadCurrentSession();
            }
            
            if (window.M) {
                M.toast({ html: 'Chat deleted', classes: 'green' });
            }
        } else {
            console.error('Failed to delete session:', data.error);
            if (window.M) {
                M.toast({ html: 'Failed to delete chat', classes: 'red' });
            }
        }
    } catch (error) {
        console.error('Error deleting session:', error);
        if (window.M) {
            M.toast({ html: 'Error deleting chat', classes: 'red' });
        }
    }
}

/**
 * Load and display current session
 */
async function loadCurrentSession() {
    try {
        const response = await fetch('/api/sessions/current');
        const data = await response.json();
        
        if (data.success) {
            activeSessionId = data.session.session_id;
            
            // Clear and render messages
            clearChatUI();
            
            if (data.messages && data.messages.length > 0) {
                renderSessionMessages(data.messages);
            } else {
                showGreeting();
            }
        }
    } catch (error) {
        console.error('Error loading current session:', error);
    }
}

/**
 * Clear the chat UI
 */
function clearChatUI() {
    const chats = document.getElementById('chats');
    if (chats) {
        chats.innerHTML = '<div class="clearfix"></div>';
    }
}

/**
 * Show greeting message
 */
function showGreeting() {
    const chats = document.getElementById('chats');
    if (!chats) return;
    
    const greeting = chats.dataset.defaultGreeting || 
        "Hello! I'm Alizha, your advanced AI assistant. How can I help you today?";
    
    const botHtml = `
        <img class="botAvatar" src="/static/images/aliza-icon.jpg" />
        <span class="botMsg">${greeting}</span>
        <div class="clearfix"></div>
    `;
    chats.insertAdjacentHTML('beforeend', botHtml);
}

/**
 * Show error in sessions list
 */
function showSessionError(message) {
    const sessionsList = document.getElementById('sessionsList');
    if (sessionsList) {
        sessionsList.innerHTML = `
            <div class="sessions-error">
                <i class="material-icons">error_outline</i>
                <p>${message}</p>
                <button class="btn-flat" onclick="loadSessions()">Retry</button>
            </div>
        `;
    }
}

/**
 * Scroll to bottom of chat
 */
function scrollToBottomOfResults() {
    const chats = document.getElementById('chats');
    if (chats) {
        chats.scrollTop = chats.scrollHeight;
    }
}

// Export functions for use in other modules
window.initSessionSidebar = initSessionSidebar;
window.loadSessions = loadSessions;
window.createNewSession = createNewSession;
window.switchToSession = switchToSession;
window.showRenameModal = showRenameModal;
window.confirmDeleteSession = confirmDeleteSession;
window.toggleSidebar = toggleSidebar;
window.openSidebar = openSidebar;
window.closeSidebar = closeSidebar;

// Initialize when DOM is ready
function initWhenReady() {
    console.log('🚀 sessionSidebar.js loaded, scheduling initialization...');
    // Delay init slightly to ensure other components are loaded
    setTimeout(() => {
        console.log('⏰ Running initSessionSidebar...');
        initSessionSidebar();
    }, 500);
}

// Check if DOM is already loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initWhenReady);
} else {
    // DOM already loaded, run directly
    initWhenReady();
}
