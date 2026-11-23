// "Search with Gemini" widget — Google Vertex AI Search (a.k.a. Agent Search / AI Applications).
//
// How it works: there is NO API key in this repo. The `configId` below points to a search app
// configured in Google Cloud Console -> AI Applications -> (app) -> Integration -> Widget tab.
// That console page controls everything: the indexed data store (a website crawl of the
// RxInfer docs sites), public access authorization, and the domain allowlist (docs.rxinfer.com
// must be allowlisted there or the widget refuses to load).
//
// If this breaks: most likely the GCP project/app behind the configId is gone or its domain
// allowlist changed. Re-create a "Website Content" data store + Search app in AI Applications
// (the app MUST be Enterprise edition - website data stores require it; keep the
// "Advanced LLM features" add-on off), grab the new configId from the Integration page,
// and update it below.
//
// Current setup (June 2026): data store "rxinfer-docs-examples" (website crawl of the
// RxInfer docs sites), Enterprise-edition search app, public access, domain allowlist
// includes docs.rxinfer.com. Shared with the RxInferExamples.jl docs (same configId there).
document.addEventListener('DOMContentLoaded', function() {
    // Create and append the search widget
    try {
        const searchWidget = document.createElement('gen-search-widget');
        searchWidget.setAttribute('configId', '25c1f1ae-0f92-4c7c-945d-5aaabdc5c7f1');
        searchWidget.setAttribute('triggerId', 'searchWidgetTrigger');
        document.body.appendChild(searchWidget);
    } catch (e) {
        console.warn('Gemini search widget failed to initialize:', e);
    }

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

        // Add "or ask DeepWiki" link (same target as the README badge)
        const deepWikiLink = document.createElement('div');
        deepWikiLink.style.cssText = `
            text-align: center;
            font-size: 0.8em;
            margin-bottom: 1rem;
        `;
        const deepWikiAnchor = document.createElement('a');
        deepWikiAnchor.textContent = 'or ask DeepWiki';
        deepWikiAnchor.setAttribute('href', 'https://deepwiki.com/ReactiveBayes/RxInfer.jl');
        deepWikiAnchor.setAttribute('target', '_blank');
        deepWikiAnchor.setAttribute('rel', 'noopener');
        deepWikiLink.appendChild(deepWikiAnchor);
        aiSearchContainer.appendChild(deepWikiLink);

        docsSearchQuery.parentNode.insertBefore(aiSearchContainer, docsSearchQuery.nextSibling);
    }

    // Load the Google Gen AI SDK
    const script = document.createElement('script');
    script.src = 'https://cloud.google.com/ai/gen-app-builder/client?hl=en_US';
    script.onerror = function() {
        console.warn('Gemini search widget: failed to load the Google Gen App Builder SDK');
    };
    document.head.appendChild(script);
});
