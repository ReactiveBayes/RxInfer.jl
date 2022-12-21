# [Publishing a new release](@id contributing-new-release)

Please first read the general [Contributing](@ref contributing-overview) section.
Also please read the [FAQ](https://github.com/JuliaRegistries/General#faq) section in the Julia General registry.

## Start the release process

In order to start the release process a person with the associated permissions should: 

- Open a commit page on GitHub
- Write the `@JuliaRegistrator register` comment for the commit:

![Release comment](../assets/img/release_comment.png)

The Julia Registrator bot should automaticallly register a requst for the new release. After all checks will be passed on the Julia Registrator side the new release will be published and tagged automatically.
