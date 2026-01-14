#!/bin/bash
# Setup pre-commit hook that auto-fixes and re-adds modified files
# This allows Claude Code to automatically fix linting issues

set -e

echo "Setting up auto-fixing pre-commit hook..."

# Create .git/hooks directory if it doesn't exist
mkdir -p .git/hooks

# Create the pre-commit hook
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
set -e
# Auto-fixing pre-commit hook for Claude Code
# Runs pre-commit and automatically adds any fixes it makes

# Run pre-commit on staged files
if ! pre-commit run 2>&1; then
    # Pre-commit failed, which could mean:
    # 1. It auto-fixed files (e.g., formatting)
    # 2. There are errors that can't be auto-fixed

    # Check if any files were modified by pre-commit
    MODIFIED_FILES=$(git diff --name-only)

    if [ -n "$MODIFIED_FILES" ]; then
        echo ""
        echo "✨ Pre-commit auto-fixed the following files:"
        echo "$MODIFIED_FILES"
        echo ""
        echo "📝 Adding fixed files to commit..."

        # Add the auto-fixed files back to staging
        echo "$MODIFIED_FILES" | xargs git add

        echo "✅ Auto-fixes applied and staged"
        echo ""

        # Check if there are still issues after auto-fix
        if ! pre-commit run --files $(git diff --cached --name-only --diff-filter=ACM) 2>&1; then
            echo ""
            echo "❌ Pre-commit still has errors that couldn't be auto-fixed."
            echo "Please review the errors above and fix them manually."
            exit 1
        fi

        echo "✅ All pre-commit checks passed after auto-fix"
    else
        echo ""
        echo "❌ Pre-commit failed with errors that couldn't be auto-fixed."
        echo "Please review the errors above and fix them manually."
        exit 1
    fi
fi

echo "✅ Pre-commit checks passed"
exit 0
EOF

# Make the hook executable
chmod +x .git/hooks/pre-commit

echo "✅ Auto-fixing pre-commit hook installed at .git/hooks/pre-commit"
echo ""
echo "How it works:"
echo "  1. When you commit, pre-commit runs automatically"
echo "  2. If pre-commit auto-fixes files (e.g., formatting), they are automatically staged"
echo "  3. If there are errors that can't be auto-fixed, the commit fails with an error message"
echo ""
