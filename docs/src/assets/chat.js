// In docs/src/assets/chat-widget.js
document.addEventListener('DOMContentLoaded', function() {
    const messenger = document.createElement('df-messenger');
    messenger.setAttribute('project-id', 'synthetic-shape-442012-s9');
    messenger.setAttribute('agent-id', '94fb46c1-e089-404f-8e0f-4478927e5755');
    messenger.setAttribute('language-code', 'en');
    messenger.setAttribute('max-query-length', '-1');

    const chatBubble = document.createElement('df-messenger-chat-bubble');
    chatBubble.setAttribute('chat-title', 'RxInfer Documentation Chatbot');
    chatBubble.setAttribute('chat-subtitle', `
        Ask questions about RxInfer Documentation
    `);
    chatBubble.setAttribute('chat-title-icon', 'assets/ld-rxinfer.png');
    messenger.appendChild(chatBubble);

    document.body.appendChild(messenger);

    const desiredWidth = 720;

    // Function to update chat window width
    const updateChatWidth = () => {
        const maxWidth = desiredWidth;
        const padding = 32; // 16px padding on each side
        const availableWidth = Math.min(window.innerWidth - padding, maxWidth);
        
        style.textContent = `
            df-messenger {
                z-index: 999;
                position: fixed;
                bottom: 16px;
                right: 16px;
                --df-messenger-font-family: 'PT Mono', system-ui, -apple-system, sans-serif;
                --df-messenger-primary-color: rgb(22, 87, 152);
                --df-messenger-focus-color: rgb(22, 87, 152);
                --df-messenger-chat-window-width: ${availableWidth}px;
                --df-messenger-chat-window-offset: 64px;
                --df-messenger-titlebar-icon-width: 64px;
                --df-messenger-titlebar-icon-height: 64px;
                --df-messenger-titlebar-icon-padding: 0 24px 0 0;
            }
        `;
    };

    // Add styles using documented properties
    const style = document.createElement('style');
    updateChatWidth(); // Initial width setting
    document.head.appendChild(style);

    // Update width on window resize
    window.addEventListener('resize', updateChatWidth);

    // Wait for custom element to be defined
    customElements.whenDefined('df-messenger').then(() => {
        setTimeout(() => {
            const chatWrapper = chatBubble.shadowRoot?.querySelector('.chat-wrapper');
            console.log(chatWrapper);
            if (chatWrapper) {
                const footer = document.createElement('div');
                const updateFooterWidth = () => {
                    const maxWidth = desiredWidth;
                    const padding = 32;
                    const availableWidth = Math.min(window.innerWidth - padding, maxWidth);
                    footer.style.cssText = `
                        font-size: 11px;
                        color: #6B7280;
                        background: #ffffff;
                        width: ${availableWidth + 2}px;
                        border-radius: 0 0 4px 4px;
                    `;
                };
                
                updateFooterWidth(); // Initial footer width
                window.addEventListener('resize', updateFooterWidth);

                const footerText = document.createElement('div');
                footerText.style.cssText = `
                    padding: 8px 16px;
                    line-height: 1.4;
                `;
                footerText.innerHTML = `
                    Responses are automatically generated and may not be accurate.
                    Please <a href="https://github.com/ReactiveBayes/RxInfer.jl/issues" target="_blank" style="color: rgb(22, 87, 152); text-decoration: none;">open an issue</a> if you find any inaccuracies.
                    Powered by Gemini.
                    Sponsored by <a href="https://lazydynamics.com" target="_blank" style="color: rgb(22, 87, 152); text-decoration: none;">Lazy Dynamics</a>.
                `;
                footer.appendChild(footerText);
                chatWrapper.appendChild(footer);
            }
        }, 10);
    });
});
