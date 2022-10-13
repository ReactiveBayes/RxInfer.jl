
// We add a simple `onload` hook to inject the custom header for our `HTML`-generated pages
window.onload = function() {
    // <header class="navigation">
    const header = document.createElement('header')
    header.classList.add("navigation")
    header.appendChild((() => {
        // <div class="container">
        const container = document.createElement('div')
        container.classList.add('container')
        container.appendChild((() => {
            // <nav class="navbar navbar-expand-lg navbar-light bg-transparent">
            const nav = document.createElement('nav')
            nav.classList.add("navbar", "navbar-expand-lg", "navbar-light", "bg-transparent")

            // Brand logo already exists in the documentation
            // nav.appendChild((() => {
            //     const brand = document.createElement('a')
            //     brand.classList.add("navbar-brand")
            //     brand.href = "https://rxinfer.ml"
            //     brand.appendChild((() => {
            //         const brandimg = document.createElement('img')
            //         brandimg.width = 100
            //         brandimg.height = '100%'
            //         brandimg.classList.add("img-fluid")
            //         brandimg.src = "assets/biglogo.svg"
            //         brandimg.alt = "RxInfer website"
            //         return brandimg
            //     })())
            //     return brand
            // })())

            nav.appendChild((() => {
                const collapse = document.createElement("div")
                collapse.classList.add("collapse", "navbar-collapse", "text-center")

                collapse.appendChild((() => {
                    const ul = document.createElement("ul")
                    ul.classList.add("navbar-nav", "mx-auto")

                    const items = [
                        { title: "Get Started", link: "poka" },
                        { title: "Documentation", link: "poka" },
                        { title: "Examples", link: "poka" },
                        { title: "Papers", link: "poka" },
                        { title: "Team", link: "poka" },
                        { title: "GitHub", link: "poka" },
                    ]

                    items.forEach((item) => {
                        ul.appendChild(((item) => {
                            const li = document.createElement("li")
                            li.classList.add("nav-item")
                            li.appendChild((() => {
                                const a = document.createElement("a")
                                a.classList.add("nav-link")
                                a.href = item.link
                                a.innerHTML = item.title
                                a.title = item.title
                                return a
                            })())
                            return li
                        })(item))
                    })
                    return ul
                })())
                return collapse
            })())
            return nav
        })())
        return container
    })())
    
    const documenterTarget = document.querySelector('#documenter');
    
    documenterTarget.parentNode.insertBefore(header, documenterTarget);
}

