
// We add a simple `onload` hook to inject the custom header for our `HTML`-generated pages
window.onload = function() {
    // <header class="navigation">
    const header = document.createElement('header')
    header.classList.add("navigation")
    header.appendChild((() => {
        const container = document.createElement('div')
        container.classList.add('container')
        container.appendChild((() => {
            const nav = document.createElement('nav')
            nav.classList.add("navbar")

            nav.appendChild((() => {
                const ul = document.createElement("ul")
                ul.classList.add("navbar-nav")

                ul.appendChild((() => {
                    const smalllink = document.createElement('li')
                    smalllink.classList.add('small-item')
                    smalllink.appendChild((() => {
                        const a = document.createElement('a')
                        a.classList.add("nav-link")
                        a.href = 'http://www.rxinfer.com'
                        a.innerHTML = 'RxInfer.jl'
                        a.title = 'RxInfer.jl'
                        return a
                    })())
                    return smalllink
                })())

                const items = [
                    { title: "Home", link: "http://www.rxinfer.com", icon: [ "fas", "fa-diagram-project" ] },
                    { title: "Get Started", link: "https://docs.rxinfer.com/stable/manuals/getting-started/", icon: [ "fas", "fa-person-chalkboard" ] },
                    { title: "Documentation", link: "https://docs.rxinfer.com/stable/", icon: [ "fas", "fa-book" ] },
                    { title: "Examples", link: "https://examples.rxinfer.com/", icon: [ "fas", "fa-laptop-code" ] },
                    { title: "Papers", link: "https://biaslab.github.io/publication/", icon: [ "far", "fa-book-open" ] },
                    { title: "Team", link: "https://github.com/orgs/ReactiveBayes/people", icon: [ "fas", "fa-people-group" ] },
                    { title: "Discussions", link: "https://github.com/orgs/ReactiveBayes/discussions", icon: [ "far", "fa-comment" ] },
                    // { title: "Contact", link: "http://www.rxinfer.com/contact/" }, the redirect is broken for now
                    { title: "GitHub", link: "https://github.com/reactivebayes/RxInfer.jl", icon: [ "fab", "fa-github" ] },
                ]

                items.forEach((item) => {
                    ul.appendChild(((item) => {
                        const li = document.createElement("li")
                        li.classList.add("nav-item")
                        li.appendChild((() => {
                            const a = document.createElement("a")

                            if (item.icon !== undefined) {
                                a.appendChild((() => {
                                    const i = document.createElement("i")
                                    i.classList.add(...(item.icon))
                                    return i    
                                })())
                            }

                            a.classList.add("nav-link")
                            a.href = item.link
                            a.title = item.title

                            a.appendChild((() => {
                                const span = document.createElement("span")
                                span.innerHTML = `&nbsp;${item.title}`
                                return span
                            })())

                            return a
                        })())
                        return li
                    })(item))
                })
                return ul
            })())
            return nav
        })())
        return container
    })())
    
    const documenterTarget = document.querySelector('#documenter');
    if (documenterTarget && documenterTarget.parentNode) {
        documenterTarget.parentNode.insertBefore(header, documenterTarget);
    }
}

document.addEventListener('DOMContentLoaded', function() {
    // === Cross-site search across external docs ===
    (function () {
        const REMOTE_SOURCES = [
            { base: 'https://examples.rxinfer.com', label: 'RxInferExamples Docs', index: null, promise: null },
            { base: 'https://reactivebayes.github.io/ReactiveMP.jl/stable', label: 'ReactiveMP Docs', index: null, promise: null },
            { base: 'https://reactivebayes.github.io/GraphPPL.jl/stable', label: 'GraphPPL Docs', index: null, promise: null },
        ];

        function esc(s) {
            return (s || '').replace(/[&<>"']/g, c =>
                ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
        }

        async function fetchRemoteIndex(source) {
            if (source.index !== null) return;
            if (source.promise) return source.promise;

            source.promise = (async () => {
                try {
                    const res = await fetch(source.base.replace(/\/+$/, '') + '/search_index.js');
                    if (!res.ok) return;
                    const text = await res.text();
                    const start = text.indexOf('{');
                    const end = text.lastIndexOf('}');
                    if (start < 0 || end < start) throw new Error('Invalid format');
                    source.index = JSON.parse(text.slice(start, end + 1)).docs || [];
                } catch (e) {
                    source.index = [];
                }
            })();

            return source.promise;
        }

        async function fetchAllRemoteIndices() {
            await Promise.all(REMOTE_SOURCES.map(fetchRemoteIndex));
        }

        function searchRemote(source, query) {
            if (!source.index || !source.index.length) return [];
            const words = query.trim().toLowerCase().split(/\s+/).filter(w => w.length > 1);
            if (!words.length) return [];
            return source.index.filter(doc => {
                const hay = (doc.title + ' ' + doc.text).toLowerCase();
                return words.every(w => hay.includes(w));
            });
        }

        function extractSnippet(text, query, contextLength = 80) {
            const words = query.trim().toLowerCase().split(/\s+/).filter(w => w.length > 1);
            if (!words.length || !text) return '';

            const lowerText = text.toLowerCase();
            let firstMatchIdx = -1;
            for (const word of words) {
                const idx = lowerText.indexOf(word);
                if (idx !== -1 && (firstMatchIdx === -1 || idx < firstMatchIdx)) {
                    firstMatchIdx = idx;
                }
            }

            if (firstMatchIdx === -1) return '';

            const start = Math.max(0, firstMatchIdx - contextLength);
            const end = Math.min(text.length, firstMatchIdx + contextLength);
            let snippet = text.slice(start, end);
            if (start > 0) snippet = '…' + snippet;
            if (end < text.length) snippet = snippet + '…';

            for (const word of words) {
                const regex = new RegExp(`(${word.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
                snippet = snippet.replace(regex, '<mark style="background-color:var(--mark-bg,#fff3cd);padding:0 2px">$1</mark>');
            }

            return snippet;
        }

        function renderSourceResults(source, results, totalCount, query) {
            const items = results.map(doc => {
                const snippet = extractSnippet(doc.text, query, 80);
                return `
            <a href="${source.base.replace(/\/+$/, '')}/${doc.location || ''}" class="search-result-link w-100 is-flex is-flex-direction-column gap-2 px-4 py-2">
                <div class="w-100 is-flex is-flex-wrap-wrap is-justify-content-space-between is-align-items-flex-start">
                    <div class="search-result-title has-text-weight-bold">${esc(doc.title)}</div>
                    <div class="property-search-result-badge">${esc(doc.category)}</div>
                </div>
                ${snippet ? `<div style="font-size:smaller;opacity:0.8;line-height:1.5">${snippet}</div>` : ''}
                <div class="has-text-left" style="font-size:smaller;opacity:0.7">
                    <i class="fas fa-external-link-alt"></i> ${source.label}: ${esc((doc.location || '').slice(0, 60))}
                </div>
            </a>
            <div class="search-divider w-100"></div>`;
            }).join('');

            const countText = totalCount > results.length
                ? `${results.length} of ${totalCount} results`
                : `${results.length} result${results.length !== 1 ? 's' : ''}`;

            return `
            <div style="padding:0.5rem 1rem;border-top:1px solid var(--card-border-color,#e9ecef);margin-top:0.5rem">
                <span class="is-size-7" style="opacity:0.7">Also from <strong>${source.label}</strong> — ${countText}</span>
            </div>
            ${items}`;
        }

        function renderResults(sections, query) {
            const html = sections.map(section =>
                renderSourceResults(section.source, section.results, section.totalCount, query)
            ).join('');
            return `<div id="cross-site-results" class="w-100 is-flex is-flex-direction-column gap-2">${html}</div>`;
        }

        let injecting = false;

        function inject() {
            if (injecting) return;
            const body = document.querySelector('.search-modal-card-body');
            const input = document.querySelector('.documenter-search-input');
            if (!body || !input || body.querySelector('#cross-site-results')) return;

            if (REMOTE_SOURCES.some(source => source.index === null)) {
                fetchAllRemoteIndices().then(() => setTimeout(inject, 0));
                return;
            }

            const query = input.value || '';
            if (query.trim().length < 2) return;

            const sections = REMOTE_SOURCES.map(source => {
                const fullResults = searchRemote(source, query);
                return {
                    source,
                    totalCount: fullResults.length,
                    results: fullResults.slice(0, 8),
                };
            }).filter(section => section.totalCount > 0);

            if (!sections.length) return;

            injecting = true;
            body.insertAdjacentHTML('beforeend', renderResults(sections, query));
            injecting = false;
        }

        let bodyObserver = null;
        function connectBodyObserver() {
            const body = document.querySelector('.search-modal-card-body');
            if (!body || bodyObserver) return;
            bodyObserver = new MutationObserver(() => { if (!injecting) setTimeout(inject, 30); });
            bodyObserver.observe(body, { childList: true });
        }

        new MutationObserver(() => {
            const modal = document.getElementById('search-modal');
            if (modal) {
                if (modal.classList.contains('is-active')) fetchAllRemoteIndices();
                connectBodyObserver();
            }
        }).observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['class'] });

        fetchAllRemoteIndices();
    })();
});

