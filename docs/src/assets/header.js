
// We add a simple `onload` hook to inject the custom header for our `HTML`-generated pages
window.onload = function() {
    const header = document.createElement('p');

    header.innerText = 'RxInfer Header Element';
    
    const documenterTarget = document.querySelector('#documenter');
    
    documenterTarget.parentNode.insertBefore(header, documenterTarget);
}

