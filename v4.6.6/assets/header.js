
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
    documenterTarget.parentNode.insertBefore(header, documenterTarget);
    
    // === Site context banner for Docs ===
    // Add banner directly to navbar after header is created
    const navbar = header.querySelector('nav.navbar');
    if (navbar) {
        const banner = document.createElement('div');
        banner.id = 'site-banner';
        banner.innerHTML = `
            📗 You are viewing the <strong>RxInfer.jl Documentation</strong>.
            Looking for comprehensive tutorials and examples? <a href="https://examples.rxinfer.com/">Go to examples website →</a>
            <button id="site-banner-close" aria-label="Close banner" title="Close banner">×</button>
        `;
        
        navbar.appendChild(banner);
        
        // Add close button handler - just hides for current page
        const closeButton = banner.querySelector('#site-banner-close');
        if (closeButton) {
            closeButton.addEventListener('click', function() {
                banner.classList.add('banner-closed');
            });
        }
    }
    
    // === Search results banner ===
    // Add banner inside modal-card-head at the bottom when search results appear
    function addSearchBanner() {
        const searchModal = document.getElementById('search-modal');
        if (!searchModal) {
            return;
        }
        
        const modalCardHead = searchModal.querySelector('.modal-card-head');
        if (!modalCardHead) {
            return;
        }
        
        // Check if banner already exists
        if (modalCardHead.querySelector('#search-results-banner')) {
            return;
        }
        
        // Create a wrapper div for the banner
        const bannerWrapper = document.createElement('div');
        bannerWrapper.style.cssText = 'width: 100%; position: absolute; bottom: 0; display: flex; justify-content: center; align-items: center;';
        bannerWrapper.id = 'search-results-banner';
        bannerWrapper.className = 'is-size-7';
        bannerWrapper.innerHTML = `
            <strong>Note:</strong>&nbsp;Search results do not include tutorials from the&nbsp;<a href="https://examples.rxinfer.com/">examples website</a>
        `;
        
        // Insert after all existing children in modal-card-head
        modalCardHead.appendChild(bannerWrapper);
    }
    
    // Watch for search modal and results to appear (search results are dynamically loaded)
    const observer = new MutationObserver(function(mutations) {
        addSearchBanner();
    });
    
    // Start observing the document body for changes
    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
    
    // Also try immediately in case search modal already exists
    setTimeout(addSearchBanner, 100);
}
