// In docs/src/assets/chat-widget.js
document.addEventListener('DOMContentLoaded', function() {
    const messenger = document.createElement('df-messenger');
    messenger.setAttribute('project-id', 'synthetic-shape-442012-s9');
    messenger.setAttribute('agent-id', '94fb46c1-e089-404f-8e0f-4478927e5755');
    messenger.setAttribute('language-code', 'en');
    messenger.setAttribute('max-query-length', '-1');

    const chatBubble = document.createElement('df-messenger-chat-bubble');
    chatBubble.setAttribute('chat-title', 'RxInfer-Assistant');
    messenger.appendChild(chatBubble);

    document.body.appendChild(messenger);

    // Add styles
    const style = document.createElement('style');
    style.textContent = `
        df-messenger {
            z-index: 999;
            position: fixed;
            --df-messenger-font-color: #2A56C6;
            --df-messenger-font-family: Roboto Mono;
            --df-messenger-chat-background: #f3f6fc;
            --df-messenger-message-user-background: #d3e3fd;
            --df-messenger-message-bot-background: #fff;
            bottom: 16px;
            right: 16px;
        }
    `;
    document.head.appendChild(style);
});