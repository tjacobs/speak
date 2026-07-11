Style guide

Functions:

- Main is the first function in the file
- Parse args is the second function in the file
- Functions go in the file in the order they are called, so reads like a book, with utils at end
- Keep main minimal, have it just call functions mainly

Comments:

- Comments are short descriptions of what the code block does, like "# Create policy", not passive "# Policy"
- Always insert a blank line immediately before every comment block, except for at top of functions
- Every block of code needs a comment above it, with a blank line above the comment
- No line of code should exist just by itself, put a comment above it or group it
- No need for blank lines at the start of functions above the first comment
- No doc strings, just put a single line comment above each function
- No parens in comments, use commas instead
- No comments at end of lines

Code:

- Allow running the program with no args to do something useful, so have defaults
- Fix or supress any warnings that occur in output, working run should be clean
- Try to not have value defaults in function define lines
- Function calls should be on one line, not broken over many lines
- Remove any imports not used

Naming:

- No variable or function name short abbreviations (e.g. ok to use config, but not cfg)
- Keep code simple and minimal number of lines
- No underscores in front of functions
- Use full variable and function names
- Prefer arg names load and save

