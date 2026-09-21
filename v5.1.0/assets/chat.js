// In docs/src/assets/chat-widget.js
document.addEventListener('DOMContentLoaded', function() {
    // Create and append the search widget
    const searchWidget = document.createElement('gen-search-widget');
    searchWidget.setAttribute('configId', '1bea2ada-cbee-4d63-9154-6e68f77e8aa0');
    searchWidget.setAttribute('triggerId', 'searchWidgetTrigger');
    document.body.appendChild(searchWidget);

    // Find the docs search query element
    const docsSearchQuery = document.getElementById('documenter-search-query');
    if (docsSearchQuery) {
        // Create container for AI search
        const aiSearchContainer = document.createElement('div');
        aiSearchContainer.style.cssText = `
            width: 14.4rem;
        `;
        aiSearchContainer.classList.add('mx-auto');
        
        // Add "or" text
        const orText = document.createElement('div');
        orText.textContent = 'or';
        orText.style.cssText = `
            text-align: center;
            color: #666;
            font-size: 0.9em;
        `;
        
        aiSearchContainer.appendChild(orText);

        // Create and append the trigger input
        const searchTrigger = document.createElement('input');
        searchTrigger.setAttribute('placeholder', 'Search with Gemini');
        searchTrigger.setAttribute('id', 'searchWidgetTrigger');
        searchTrigger.classList.add('docs-search-query','input','is-rounded','is-small','is-clickable','my-2','py-1','px-2');
        
        aiSearchContainer.appendChild(searchTrigger);
        docsSearchQuery.parentNode.insertBefore(aiSearchContainer, docsSearchQuery.nextSibling);

        
    }

    // Load the Google Gen AI SDK
    const script = document.createElement('script');
    script.src = 'https://cloud.google.com/ai/gen-app-builder/client?hl=en_US';
    document.head.appendChild(script);
});
