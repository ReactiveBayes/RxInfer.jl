# [Publishing a new release](@id contributing-new-release)

Please read first the general [Contributing](@ref contributing-overview) section.
Also, please read the [FAQ](https://github.com/JuliaRegistries/General#faq) section in the Julia General registry.

## Start the release process

In order to start the release process a person with the necessary permissions should:

- Open a commit page on GitHub
- Write the `@JuliaRegistrator register` comment for the commit:

![Release comment](../assets/img/release_comment.png)

The Julia Registrator bot should automatically register a request for the new release. Once all checks have passed on the Julia Registrator's side, the new release will be published and tagged automatically.
