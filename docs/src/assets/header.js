
// We add a simple `onload` hook to inject the custom header for our `HTML`-generated pages
window.onload = function() {
    const header = document.createElement('div');
    const link = document.createElement('a')

    link.href = "https://github.com/biaslab/RxInfer.jl"
    link.innerText = "Open GitHub repository"

    header.classList.add('rx-infer-header')
    header.appendChild(link)
    
    const documenterTarget = document.querySelector('#documenter');
    
    documenterTarget.parentNode.insertBefore(header, documenterTarget);
}

