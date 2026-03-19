---
name: test_throws with error messages
description: User prefers using @test_throws with partial string matching for error messages, not just exception types
type: feedback
---

When writing `@test_throws` tests, prefer matching on the error message string (partial match) rather than just the exception type. This gives better test coverage by verifying the actual error content.

**Why:** The user wants tests to verify the actual error message, not just that some error was thrown.

**How to apply:** Use `@test_throws "partial error message" expr` style instead of `@test_throws ErrorType expr` when the error message is known and meaningful.
