# Developer Notes

## Precompiled Traces

The CI workflow automatically generates compilation traces during test runs. These traces contain information about which functions are compiled during the test suite execution and can be useful for optimizing package loading time and first-time run latency.

The traces are saved as artifacts in GitHub Actions and are retained for 30 days. You can download these artifacts from the workflow run page to analyze package compilation patterns or use them for precompilation.

To access the traces:
1. Go to the GitHub Actions page for the repository
2. Select a completed workflow run
3. Scroll down to the Artifacts section
4. Download the `server-compilation-trace-*.jl` file

These traces can be used in combination with `PackageCompiler.jl`. See the documentation for PackageCompiler for more details.

## Test Output Artifacts

The CI workflow also saves test output artifacts that contain the results of test runs. These artifacts include generated plots, logs, and other outputs produced during test execution.

Like the compilation traces, test output artifacts are retained for 30 days and can be accessed from the GitHub Actions workflow run page. They are named `test-output-<version>-<os>-<arch>` and can be helpful for debugging test failures or examining test behavior across different Julia versions and operating systems.

To use these artifacts:
1. Navigate to the GitHub Actions workflow run
2. Download the test output artifacts
3. Extract the contents to examine the test-generated files
