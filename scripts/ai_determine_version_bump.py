#!/usr/bin/env python3
"""
Use AI (Gemini, Claude, or OpenAI) to analyze git commits and determine semantic version bump type.

According to CONTRIBUTING.md:
- major increment: incompatible API changes
- minor increment: added functionality that is backwards-compatible
- patch increment: backwards-compatible bug fixes

Requires one of the following environment variables to be set:
- GEMINI_API_KEY for Google Gemini
- ANTHROPIC_API_KEY for Claude
- OPENAI_API_KEY for OpenAI

Optional environment variables:
- OPENAI_MODEL (default: gpt-4o) - specify which OpenAI model to use
- CLAUDE_MODEL (default: claude-3-5-sonnet-20241022) - specify which Claude model to use
- GEMINI_MODEL (default: gemini-2.0-flash-exp) - specify which Gemini model to use

The script will try providers in order: OpenAI -> Claude -> Gemini -> Fallback

Install: pip install openai anthropic google-generativeai
"""

import argparse
import json
import os
import re
import subprocess
import sys
from typing import Tuple


def get_latest_tag() -> str:
    """Get the latest git tag."""
    try:
        result = subprocess.run(
            ["git", "tag", "--sort=-v:refname"],
            capture_output=True,
            text=True,
            check=True,
        )
        tags = [
            line.strip() for line in result.stdout.strip().split("\n") if line.strip()
        ]
        if tags:
            return tags[0]
        return ""
    except subprocess.CalledProcessError:
        return ""


def get_commits_since_tag(tag: str) -> list[dict]:
    """Get commit messages and diffs since the given tag."""
    try:
        if tag:
            cmd = [
                "git",
                "log",
                f"{tag}..HEAD",
                "--pretty=format:%H|||%s|||%b",
            ]
        else:
            cmd = ["git", "log", "--pretty=format:%H|||%s|||%b"]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        commits = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split("|||")
            if len(parts) >= 2:
                commit_hash = parts[0]
                subject = parts[1]
                body = parts[2] if len(parts) > 2 else ""

                # Get file changes for this commit
                diff_cmd = ["git", "diff", f"{commit_hash}^..{commit_hash}", "--stat"]
                diff_result = subprocess.run(diff_cmd, capture_output=True, text=True)
                files_changed = diff_result.stdout.strip()

                commits.append(
                    {
                        "hash": commit_hash[:7],
                        "subject": subject,
                        "body": body,
                        "files_changed": files_changed,
                    }
                )

        return commits
    except subprocess.CalledProcessError:
        return []


def analyze_with_openai(commits_summary: str, current_version: str) -> dict:
    """Use OpenAI to analyze commits."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")

    model = os.getenv("OPENAI_MODEL", "gpt-4o")

    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    prompt = f"""You are analyzing git commits for a CUDA kernel library called FlashInfer to determine the appropriate semantic version bump.

Current version: {current_version}

Semantic versioning rules for this project (from CONTRIBUTING.md):
- MAJOR increment: incompatible API changes (breaking changes to public APIs)
- MINOR increment: added functionality that is backwards-compatible (new kernels, new features, new SM support, etc.)
- PATCH increment: backwards-compatible bug fixes (both functional and performance fixes)

Here are the commits since the last release:

{commits_summary}

Please analyze these commits and determine:
1. Whether there are any breaking API changes (MAJOR bump needed)
2. Whether there are new features or backwards-compatible functionality additions (MINOR bump needed)
3. Whether there are only bug fixes without new features (PATCH bump needed)
4. If no significant changes, return "none"

Respond in JSON format:
{{
    "bump_type": "major|minor|patch|none",
    "reasoning": "Detailed explanation of your decision",
    "key_changes": ["list of most important changes that influenced the decision"]
}}

Important considerations:
- Internal refactoring, test updates, documentation changes alone don't warrant a version bump
- Performance improvements are considered bug fixes (PATCH)
- New kernel implementations or new features are MINOR bumps
- API signature changes or removed functionality are MAJOR bumps
- Focus on changes that affect users of the library, not internal changes
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that analyzes git commits to determine semantic version bumps. Always respond with valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        response_format={"type": "json_object"},
    )

    result_text = response.choices[0].message.content.strip()
    return json.loads(result_text)


def analyze_with_claude(commits_summary: str, current_version: str) -> dict:
    """Use Anthropic Claude to analyze commits."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    model = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")

    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)

    prompt = f"""You are analyzing git commits for a CUDA kernel library called FlashInfer to determine the appropriate semantic version bump.

Current version: {current_version}

Semantic versioning rules for this project (from CONTRIBUTING.md):
- MAJOR increment: incompatible API changes (breaking changes to public APIs)
- MINOR increment: added functionality that is backwards-compatible (new kernels, new features, new SM support, etc.)
- PATCH increment: backwards-compatible bug fixes (both functional and performance fixes)

Here are the commits since the last release:

{commits_summary}

Please analyze these commits and determine:
1. Whether there are any breaking API changes (MAJOR bump needed)
2. Whether there are new features or backwards-compatible functionality additions (MINOR bump needed)
3. Whether there are only bug fixes without new features (PATCH bump needed)
4. If no significant changes, return "none"

Respond in JSON format:
{{
    "bump_type": "major|minor|patch|none",
    "reasoning": "Detailed explanation of your decision",
    "key_changes": ["list of most important changes that influenced the decision"]
}}

Important considerations:
- Internal refactoring, test updates, documentation changes alone don't warrant a version bump
- Performance improvements are considered bug fixes (PATCH)
- New kernel implementations or new features are MINOR bumps
- API signature changes or removed functionality are MAJOR bumps
- Focus on changes that affect users of the library, not internal changes
"""

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}],
    )

    result_text = response.content[0].text.strip()

    # Extract JSON from response (might be wrapped in markdown code blocks)
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", result_text, re.DOTALL)
    if json_match:
        result_text = json_match.group(1)

    return json.loads(result_text)


def analyze_with_gemini(commits_summary: str, current_version: str) -> dict:
    """Use Google Gemini to analyze commits."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")

    import google.generativeai as genai

    genai.configure(api_key=api_key)

    prompt = f"""You are analyzing git commits for a CUDA kernel library called FlashInfer to determine the appropriate semantic version bump.

Current version: {current_version}

Semantic versioning rules for this project (from CONTRIBUTING.md):
- MAJOR increment: incompatible API changes (breaking changes to public APIs)
- MINOR increment: added functionality that is backwards-compatible (new kernels, new features, new SM support, etc.)
- PATCH increment: backwards-compatible bug fixes (both functional and performance fixes)

Here are the commits since the last release:

{commits_summary}

Please analyze these commits and determine:
1. Whether there are any breaking API changes (MAJOR bump needed)
2. Whether there are new features or backwards-compatible functionality additions (MINOR bump needed)
3. Whether there are only bug fixes without new features (PATCH bump needed)
4. If no significant changes, return "none"

Respond in JSON format:
{{
    "bump_type": "major|minor|patch|none",
    "reasoning": "Detailed explanation of your decision",
    "key_changes": ["list of most important changes that influenced the decision"]
}}

Important considerations:
- Internal refactoring, test updates, documentation changes alone don't warrant a version bump
- Performance improvements are considered bug fixes (PATCH)
- New kernel implementations or new features are MINOR bumps
- API signature changes or removed functionality are MAJOR bumps
- Focus on changes that affect users of the library, not internal changes
"""

    model = genai.GenerativeModel(model_name)

    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.3,
        ),
    )

    result_text = response.text.strip()

    # Extract JSON from response (might be wrapped in markdown code blocks)
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", result_text, re.DOTALL)
    if json_match:
        result_text = json_match.group(1)

    return json.loads(result_text)


def analyze_with_ai(commits: list[dict], current_version: str) -> dict:
    """
    Use AI to analyze commits and determine version bump.

    Tries providers in order: OpenAI -> Claude -> Gemini -> Fallback
    """
    # Prepare the commits summary once
    commits_summary = "\n\n".join(
        [
            f"Commit {c['hash']}:\n"
            f"Subject: {c['subject']}\n"
            f"Body: {c['body']}\n"
            f"Files changed:\n{c['files_changed'][:500]}"  # Limit file changes to avoid token limit
            for c in commits
        ]
    )

    # Try OpenAI first
    try:
        print("Trying OpenAI...", file=sys.stderr)
        result = analyze_with_openai(commits_summary, current_version)
        print(
            f"Successfully used OpenAI (model: {os.getenv('OPENAI_MODEL', 'gpt-4o')})",
            file=sys.stderr,
        )
        return result
    except ImportError:
        print(
            "OpenAI package not installed. Install with: pip install openai",
            file=sys.stderr,
        )
    except ValueError as e:
        print(f"OpenAI not available: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Error calling OpenAI API: {e}", file=sys.stderr)

    # Try Claude second
    try:
        print("Trying Anthropic Claude...", file=sys.stderr)
        result = analyze_with_claude(commits_summary, current_version)
        print(
            f"Successfully used Anthropic Claude (model: {os.getenv('CLAUDE_MODEL', 'claude-3-5-sonnet-20241022')})",
            file=sys.stderr,
        )
        return result
    except ImportError:
        print(
            "Anthropic package not installed. Install with: pip install anthropic",
            file=sys.stderr,
        )
    except ValueError as e:
        print(f"Claude not available: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Error calling Claude API: {e}", file=sys.stderr)

    # Try Gemini third
    try:
        print("Trying Google Gemini...", file=sys.stderr)
        result = analyze_with_gemini(commits_summary, current_version)
        print(
            f"Successfully used Google Gemini (model: {os.getenv('GEMINI_MODEL', 'gemini-2.0-flash-exp')})",
            file=sys.stderr,
        )
        return result
    except ImportError:
        print(
            "Gemini package not installed. Install with: pip install google-generativeai",
            file=sys.stderr,
        )
    except ValueError as e:
        print(f"Gemini not available: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Error calling Gemini API: {e}", file=sys.stderr)

    # Fallback to basic analysis
    print(
        "Warning: No AI providers available, falling back to basic analysis",
        file=sys.stderr,
    )
    return fallback_analysis(commits)


def fallback_analysis(commits: list[dict]) -> dict:
    """
    Fallback analysis using simple keyword matching.
    """
    has_major = False
    has_minor = False
    has_patch = False
    key_changes = []

    for commit in commits:
        text = (commit["subject"] + " " + commit["body"]).lower()

        # Skip version bump commits
        if re.search(r"bump version|release.*v?\d+\.\d+\.\d+", text):
            continue

        # Check for breaking changes
        if any(
            keyword in text
            for keyword in [
                "breaking change",
                "break:",
                "breaking:",
                "!:",
                "incompatible",
            ]
        ):
            has_major = True
            key_changes.append(f"Breaking change: {commit['subject']}")

        # Check for features
        elif any(
            keyword in text
            for keyword in ["feat:", "feature:", "add ", "implement", "support "]
        ):
            has_minor = True
            key_changes.append(f"New feature: {commit['subject']}")

        # Check for fixes
        elif any(keyword in text for keyword in ["fix:", "bugfix:", "fix ", "fixes "]):
            has_patch = True
            key_changes.append(f"Bug fix: {commit['subject']}")

    if has_major:
        bump_type = "major"
        reasoning = "Detected breaking changes in commits"
    elif has_minor:
        bump_type = "minor"
        reasoning = "Detected new features without breaking changes"
    elif has_patch:
        bump_type = "patch"
        reasoning = "Detected bug fixes only"
    else:
        bump_type = "none"
        reasoning = "No significant changes detected (chore, docs, tests only)"

    return {
        "bump_type": bump_type,
        "reasoning": reasoning,
        "key_changes": key_changes[:10],  # Limit to top 10
    }


def parse_version(version_str: str) -> Tuple[int, int, int, str]:
    """Parse version string like '0.4.1' or '0.4.0rc1'."""
    version_str = version_str.lstrip("v")
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)(.*)$", version_str)
    if not match:
        raise ValueError(f"Invalid version format: {version_str}")

    major, minor, patch, suffix = match.groups()
    return int(major), int(minor), int(patch), suffix


def bump_version(current_version: str, bump_type: str) -> str:
    """Bump the version according to semantic versioning."""
    major, minor, patch, suffix = parse_version(current_version)

    if bump_type == "major":
        return f"{major + 1}.0.0"
    elif bump_type == "minor":
        return f"{major}.{minor + 1}.0"
    elif bump_type == "patch":
        return f"{major}.{minor}.{patch + 1}"
    else:
        raise ValueError(f"Invalid bump type: {bump_type}")


def main():
    parser = argparse.ArgumentParser(
        description="Use AI to determine semantic version bump type based on git commits"
    )
    parser.add_argument(
        "--current-version",
        help="Current version (if not provided, will read from version.txt)",
    )
    parser.add_argument(
        "--since-tag",
        help="Analyze commits since this tag (if not provided, uses latest tag)",
    )
    parser.add_argument(
        "--output-format",
        choices=["simple", "json"],
        default="simple",
        help="Output format",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed analysis",
    )

    args = parser.parse_args()

    # Get current version
    if args.current_version:
        current_version = args.current_version
    else:
        try:
            with open("version.txt", "r") as f:
                current_version = f.read().strip()
        except FileNotFoundError:
            print(
                "Error: version.txt not found and --current-version not provided",
                file=sys.stderr,
            )
            sys.exit(1)

    # Get commits to analyze
    if args.since_tag:
        tag = args.since_tag
    else:
        tag = get_latest_tag()

    if args.verbose:
        print(f"Current version: {current_version}", file=sys.stderr)
        print(
            f"Analyzing commits since: {tag if tag else 'beginning'}", file=sys.stderr
        )

    commits = get_commits_since_tag(tag)

    if not commits:
        if args.verbose:
            print("No commits to analyze", file=sys.stderr)

        if args.output_format == "json":
            print(json.dumps({"bump_type": "none", "new_version": current_version}))
        else:
            print("none")
        sys.exit(0)

    if args.verbose:
        print(f"\nAnalyzing {len(commits)} commits with AI...", file=sys.stderr)

    # Use AI to analyze
    result = analyze_with_ai(commits, current_version)

    bump_type = result.get("bump_type", "none")
    reasoning = result.get("reasoning", "")
    key_changes = result.get("key_changes", [])

    # Calculate new version
    if bump_type == "none":
        new_version = current_version
    else:
        new_version = bump_version(current_version, bump_type)

    if args.verbose:
        print("\n=== AI Analysis Result ===", file=sys.stderr)
        print(f"Bump type: {bump_type}", file=sys.stderr)
        print(f"New version: {new_version}", file=sys.stderr)
        print(f"\nReasoning: {reasoning}", file=sys.stderr)
        if key_changes:
            print("\nKey changes:", file=sys.stderr)
            for change in key_changes:
                print(f"  - {change}", file=sys.stderr)

    # Output result
    if args.output_format == "json":
        output = {
            "bump_type": bump_type,
            "current_version": current_version,
            "new_version": new_version,
            "reasoning": reasoning,
            "key_changes": key_changes,
        }
        print(json.dumps(output, indent=2))
    else:
        # Simple format: "bump_type new_version"
        print(f"{bump_type} {new_version}")


if __name__ == "__main__":
    main()
